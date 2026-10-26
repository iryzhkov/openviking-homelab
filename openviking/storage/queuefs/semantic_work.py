# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Per-message semantic lifecycle, including Skill update cancellation and locks."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Awaitable, Callable, TypeVar

from openviking.server.identity import RequestContext
from openviking.service.task_tracker import get_task_tracker
from openviking.service.task_tracker_concurrency import run_to_completion
from openviking.service.task_work_index import get_enqueued_work_ids, get_task_context
from openviking.storage.errors import LockAcquisitionError
from openviking.storage.queuefs.process_result import ProcessResult
from openviking.storage.queuefs.semantic_executor import SemanticTreeStats
from openviking.storage.queuefs.semantic_lock import ResourceInputConflict, SemanticLockScope
from openviking.storage.queuefs.semantic_msg import SemanticMsg
from openviking.storage.viking_fs import get_viking_fs
from openviking.telemetry import (
    OperationTelemetry,
    register_telemetry,
    resolve_telemetry,
    unregister_telemetry,
)
from openviking.utils.content_hash import content_md5

if TYPE_CHECKING:
    from openviking.storage.queuefs.semantic_processor import SemanticProcessor

_Result = TypeVar("_Result")


class SemanticMessageWork:
    """Original Resource/Memory lifecycle; each dequeued message owns its scope."""

    def __init__(
        self,
        processor: SemanticProcessor,
        msg: SemanticMsg,
        caller_lock: dict[str, Any] | None,
    ) -> None:
        self.processor = processor
        self.msg = msg
        self.caller_lock = caller_lock
        self.scope: SemanticLockScope | None = None
        self._lock_closed = False

    def start(self) -> None:
        pass

    def resolve_telemetry(self) -> OperationTelemetry | None:
        return resolve_telemetry(self.msg.telemetry_id)

    async def acquire_lock(self, ctx: RequestContext) -> bool:
        async def acquire() -> None:
            self.scope = await SemanticLockScope.resolve(
                self.msg.lock_handoff,
                caller_lock=self.caller_lock,
                fallback_path_factory=lambda: get_viking_fs()._uri_to_path(self.msg.uri, ctx=ctx),
            )

        await run_to_completion(acquire)
        return True

    async def run_write(self, factory: Callable[[], Awaitable[_Result]]) -> _Result:
        return await factory()

    async def finish_processing(self, succeeded: bool) -> None:
        assert self.scope is not None
        if self.msg.lock_handoff is not None:
            await self._wait_for_embedding_work(
                stop_waiting=getattr(self.processor, "_embedding_worker_stopped", None),
            )
        if succeeded:
            await self.scope.close()
            self._lock_closed = True

    def failure_result(self, stats: SemanticTreeStats | None) -> ProcessResult | None:
        return None

    async def _wait_for_embedding_work(
        self,
        *,
        stop_waiting: Callable[[], bool] | None = None,
    ) -> None:
        context = get_task_context()
        work_ids = get_enqueued_work_ids("Embedding")
        if context is not None and work_ids:
            await get_task_tracker().wait_for_work(
                context.task_id,
                work_ids,
                stop_waiting=stop_waiting,
            )

    async def reenqueue(self) -> None:
        async def enqueue(queue, msg):
            if self.scope is not None and self.scope.lock is not None and self.scope._owned:
                await self.processor._enqueue_semantic_retry(queue, msg, self.scope)
            else:
                await queue.enqueue(msg)

        await self.processor._reenqueue_semantic_msg(self.msg, enqueue=enqueue)

    async def reenqueue_after_error(self) -> None:
        await self.reenqueue()

    async def skip(self) -> None:
        if self.msg.lock_handoff is not None:
            await self.processor._release_cancelled_semantic_lock(self.msg)

    async def cancel(self) -> None:
        if self.scope is None:
            await self.skip()

    async def cancel_queued(self) -> None:
        await self.processor._release_cancelled_semantic_lock(self.msg)

    async def close(self) -> None:
        if self.scope is not None and not self._lock_closed:
            await self.scope.close()
            self._lock_closed = True


class ResourceSemanticMessageWork(SemanticMessageWork):
    """Fence resource inputs with continuously owned tree coverage."""

    async def acquire_lock(self, ctx: RequestContext) -> bool:
        async def acquire() -> None:
            fs = get_viking_fs()
            msg = self.msg
            from openviking.server.account_stores.models import AccountIdentity
            from openviking.service.account_identity import recheck_account_identity

            # Only the original authorized request may supply the identity.
            # Refuse before any lock or filesystem mutation; never capture a
            # fresh identity for a cached message.
            original = msg.resource_creation_identity
            if not isinstance(original, str) or not original:
                raise ResourceInputConflict(
                    "Resource semantic work lacks its original request identity"
                )
            admitted = AccountIdentity(original, False)
            await recheck_account_identity(fs._account_identity_manager, ctx.account_id, admitted)
            self.scope = await SemanticLockScope.resolve(
                msg.lock_handoff, caller_lock=self.caller_lock, refuse_stale=True
            )
            if self.scope.lock is None:
                # Resource-root coverage includes every recursive effect.
                lease = await fs._async_agfs.pathlock_acquire_batch(
                    [{"path": fs._uri_to_path("viking://resources", ctx=ctx), "kind": "tree"}]
                )
                self.scope = SemanticLockScope(lease, _owned=True)
            required = [
                msg.uri,
                msg.target_uri or msg.uri,
                *msg.file_md5s,
                *msg.file_abstracts,
                *msg.file_vector_actions,
                *msg.file_ingest_options,
            ]
            # The opaque lease descriptor intentionally omits coverage. Export
            # it from the manager using the current ownership capability.
            owner = self.caller_lock if self.caller_lock is not None else self.scope.lock
            current = await fs._async_agfs.pathlock_to_handoff(owner)
            covered = current.get("covered_paths") or []
            for uri in required:
                path = fs._uri_to_path(uri, ctx=ctx).rstrip("/")
                if not any(
                    item.get("kind") == "tree"
                    and (
                        path == item.get("path", "").rstrip("/")
                        or path.startswith(item.get("path", "").rstrip("/") + "/")
                    )
                    for item in covered
                ):
                    raise ResourceInputConflict("Resource semantic effects lack tree coverage")
            if msg.lock_handoff is None:
                # Plans, staging copies and parse artifacts contain historical
                # material without a complete compare condition in this schema.
                if (
                    msg.plan is not None
                    or msg.artifact_ref
                    or msg.artifact_files
                    or msg.source
                    or msg.copy_source_uri
                    or (msg.target_uri and msg.target_uri != msg.uri)
                ):
                    raise ResourceInputConflict("Resource cached inputs require a live handoff")
                cached_files = (
                    set(msg.file_abstracts)
                    | set(msg.file_vector_actions)
                    | set(msg.file_ingest_options)
                )
                if msg.generation_trigger == "content_write":
                    cached_files.update(
                        uri
                        for kind in ("added", "modified")
                        for uri in (msg.changes or {}).get(kind, [])
                    )
                if not cached_files.issubset(msg.file_md5s):
                    raise ResourceInputConflict(
                        "Resource cached inputs lack byte compare conditions"
                    )
                for uri, expected in msg.file_md5s.items():
                    raw = await fs.read_file_bytes(uri, ctx=ctx)
                    if not expected or content_md5(raw) != expected:
                        raise ResourceInputConflict(
                            "Resource input bytes changed before processing"
                        )
            if not msg.skip_vectorization and get_task_context() is None:
                raise ResourceInputConflict("Resource vector work requires tracked completion")
            from openviking.storage.queuefs.resource_embedding_fence import (
                ResourceEmbeddingProducer,
            )

            # Compare the original identity again after final coverage.
            await recheck_account_identity(fs._account_identity_manager, ctx.account_id, admitted)
            # Processing-time identity does not prove the original request
            # lifetime, so cached plan patches/deletes stay fail-closed here.
            self._embedding_producer = ResourceEmbeddingProducer(ctx, admitted, owner)

        await run_to_completion(acquire)
        from openviking.storage.queuefs.resource_embedding_fence import (
            bind_resource_embedding_producer,
        )

        self._producer_binding = bind_resource_embedding_producer(self._embedding_producer)
        self._producer_binding.__enter__()
        return True

    async def validate_ownership(self) -> None:
        assert self.scope is not None and self.scope.lock is not None
        owner = self.caller_lock if self.caller_lock is not None else self.scope.lock
        try:
            status = await get_viking_fs()._async_agfs.pathlock_refresh(owner)
        except (LockAcquisitionError, RuntimeError, ValueError) as error:
            raise ResourceInputConflict("Resource lease ownership was lost") from error
        if status != "refreshed":
            raise ResourceInputConflict("Resource lease ownership was lost")

    async def run_write(self, factory: Callable[[], Awaitable[_Result]]) -> _Result:
        async def write() -> _Result:
            await self.validate_ownership()
            return await factory()

        return await run_to_completion(write)

    async def finish_processing(self, succeeded: bool) -> None:
        # Resource writes enqueue vectors even without a producer handoff.
        # Drain started work on failure/cancellation as well as success. The
        # embedding consumer still needs its own final fence after ownership
        # loss or process restart; a wait here cannot qualify that boundary.
        if not self.msg.skip_vectorization:
            await run_to_completion(self._wait_for_embedding_work)
        if succeeded:
            assert self.scope is not None
            await self.scope.close()
            self._lock_closed = True

    async def close(self) -> None:
        try:
            await super().close()
        except (LockAcquisitionError, RuntimeError, ValueError) as error:
            # Release was attempted with our original capability. A revoked
            # capability cannot release a new owner; never reacquire or remove
            # directories to clean up that owner's work.
            if "owned lease capability does not match ref" not in str(error):
                raise
            assert self.scope is not None
            self.scope.lock = None
            self.scope._owned = False
            self._lock_closed = True
        finally:
            binding = getattr(self, "_producer_binding", None)
            if binding is not None:
                self._producer_binding = None
                binding.__exit__(None, None, None)

    def failure_result(self, stats: SemanticTreeStats | None) -> ProcessResult | None:
        if stats is not None and stats.failures:
            self.processor._merge_request_stats(
                self.msg.telemetry_id, error_count=len(stats.failures)
            )
            return ProcessResult.failed("\n".join(stats.failures), errors=tuple(stats.failures))
        return None


class SkillSemanticMessageWork(SemanticMessageWork):
    """Keep Skill tracking and package ownership alive until writes settle.

    A failed run retains its lease for retry; cancellation drains started work
    before update rollback can proceed. None of this state is shared by messages.
    """

    def __init__(
        self,
        processor: SemanticProcessor,
        msg: SemanticMsg,
        caller_lock: dict[str, Any] | None,
    ) -> None:
        super().__init__(processor, msg, caller_lock)
        self._owns_telemetry = False
        self._lock_started = False
        self._lock_closed = False

    def start(self) -> None:
        msg = self.msg
        if not msg.telemetry_id:
            msg.telemetry_id = OperationTelemetry(
                operation="skill_index", enabled=False
            ).telemetry_id

    def resolve_telemetry(self) -> OperationTelemetry:
        collector = super().resolve_telemetry()
        if collector is None:
            collector = OperationTelemetry(operation="skill_index", enabled=False)
            collector.telemetry_id = self.msg.telemetry_id
            register_telemetry(collector)
            self._owns_telemetry = True
        return collector

    async def acquire_lock(self, ctx: RequestContext) -> bool:
        self._lock_started = True
        self.scope = await self.processor._resolve_skill_semantic_lock(
            self.msg, ctx, self.caller_lock
        )
        return self.scope.lock is not None

    async def run_write(self, factory: Callable[[], Awaitable[_Result]]) -> _Result:
        return await run_to_completion(factory)

    async def finish_processing(self, succeeded: bool) -> None:
        async def finish():
            if not self.msg.skip_vectorization:
                await self._wait_for_embedding_work(
                    stop_waiting=self.processor._embedding_worker_stopped,
                )
            if succeeded:
                assert self.scope is not None
                await self.scope.close()
                self._lock_closed = True

        # On failure retain the lease until reenqueue hands it to the retry.
        await run_to_completion(finish)

    def failure_result(self, stats: SemanticTreeStats | None) -> ProcessResult | None:
        if stats is None or not stats.failures:
            return None
        # Report every file error, but settle this queue item only once.
        self.processor._merge_request_stats(self.msg.telemetry_id, error_count=len(stats.failures))
        return ProcessResult.failed(
            "\n".join(stats.failures),
            errors=tuple(stats.failures),
        )

    async def reenqueue(self) -> None:
        async def enqueue(queue, msg):
            if self.scope is not None and self.scope.lock is not None and self.scope._owned:
                await run_to_completion(
                    lambda: self.processor._enqueue_skill_retry(queue, msg, self.scope)
                )
            else:
                await run_to_completion(lambda: queue.enqueue(msg))

        # The shared circuit-breaker delay remains interruptible. Only the
        # actual lock handoff and enqueue operation must finish once started.
        await self.processor._reenqueue_semantic_msg(self.msg, enqueue=enqueue)

    async def reenqueue_after_error(self) -> None:
        try:
            await self.reenqueue()
        except asyncio.CancelledError:
            await run_to_completion(
                lambda: self.processor._release_cancelled_semantic_lock(self.msg)
            )
            raise

    async def skip(self) -> None:
        if self.msg.lock_handoff is not None:
            await run_to_completion(
                lambda: self.processor._release_cancelled_semantic_lock(self.msg)
            )

    async def cancel(self) -> None:
        if not self._lock_started:
            # Cancellation during the circuit-breaker delay precedes adoption.
            await self.skip()

    async def close(self) -> None:
        if self.scope is not None and not self._lock_closed:
            await run_to_completion(self.scope.close)
        if self._owns_telemetry:
            unregister_telemetry(self.msg.telemetry_id)

    async def cancel_queued(self) -> None:
        # Skill rollback must observe release before this item is settled.
        await self.processor._release_cancelled_semantic_lock(self.msg)
