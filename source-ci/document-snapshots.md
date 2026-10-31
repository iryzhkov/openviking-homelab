# Focused document snapshot qualification

`document-snapshots.yml` exports the checked-out source without Git metadata and runs `run_document_snapshot_checks.sh`. The script is the exact harness that passed the 20 document-write and 30 reindex effect-bounds cases in Chaldea at product commit `9a64acd7756f2611cef4cd35221e679cfabf115d`.

The harness pins eight source/test Git blobs, verifies the anonymous dependency image and every layer, overlays and compares the current Python source and tests, and records native hashes and module origins. A future change to those eight files needs an explicit pin update and fresh acceptance. The fixture uses the ordinary native local storage factory, synthetic model providers, and real queue/tracker functions. It omits only the existing mock-only SessionCommit fallback. Tests cannot connect to external services, and all writes use asserted disk scratch HOME/TMPDIR.

The CI report gate requires exactly 35 document and 30 reindex cases with zero errors, failures, skips or expected failures, even when pytest itself exits zero. Its separate acceptance tests exercise passing, skipped/expected-failure, failed/error, missing-case and missing-report inputs.

The 65 cases include exact directory replacement at the original URI and cover snapshot/backup/rollback, restricted outsider access, direct control-target and reindex refusal, canonical identity, and reindex effect boundaries. The current regression's baseline failed during `document_node` on `.relations.json`, before replacement/backup; the separate baseline job's overly strict post-validator failed too, so that job is not reported as passing.

## Verified document ACK

A document-write success may return `content_state=committed`,
`processing_state=unobserved`, and `cleanup_state=unknown` in either existing
contract version. Bytes, controls, and any requested backup have been verified
under the original lease. Capture, queue handoff, and context cleanup continue
in that same transaction, held strongly by FSService. The ACK is a deep copy.
A completed transaction wins the ACK race and returns its full result instead.
Caller cancellation before ACK waits for the transaction and then propagates
without returning an ACK.

Service close drains these transactions without cancelling them before closing
queue, background-task, and tracker dependencies. App lifespan also drains while
gateway and MCP contexts are active. Existing capture/handoff/enqueue exceptions
remain caught and leave processing unobserved; they now log only a fixed operation
and error class. Escaping continuation errors are observed and logged once by the
task owner without exception text or traceback.

This success means committed and nonretryable. It does not prove queue, indexing,
or cleanup completion, and it does not waive exact content, archive, backup-read,
or retrieval gates. No timeout, JWT expiry, or contract version changes.

FSService admits at most 32 owned document transactions with one fixed bounded
semaphore. Admission is cancellable and precedes coroutine/task creation or mutation.
Only the task done callback returns the permit, including on failure; ACK does not.
The URI coordinator and original lease remain held through transaction exit.
Native account coverage excludes a competing same-URI write with an immediate
LockAcquisitionError before mutation; it does not promise to wait for the coordinator.
This bounds owned transactions, including continuations, but not incoming admission
waiters. A global request admission limit remains an existing gap.

The added event-gated cases require the actual enqueue/capture functions after
their gates, with native queue/tracker methods unpatched. They cover ownership,
readable bytes and backup, cleanup, precommit failure, cancellation, shutdown
ordering, ASGI response delivery, serialization, caught and escaping postcommit
failures, and the completed-task race. These new cases are pending validation:
the first Chaldea submission failed locally during snapshot creation with ELOOP.
The updated 35/30 report gate is a required acceptance target, not a passing claim.

This change addresses the source-proven postcommit delay class only. The incident
cause remains unproven; the existing hold and paused job remain untouched.

The native libraries come from a digest-verified dependency image and were not rebuilt from this source revision. Test execution does not prove the held live OV note has a relations sidecar, that deployment resolves its refusal, or that the curator pilot or scheduling qualifies. The historical filter-only `source-checks.yml` attestation remains unchanged and is not applicable to this batch.
