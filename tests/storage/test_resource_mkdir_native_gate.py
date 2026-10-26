"""Native qualification gate: expected RED with the old image binary.

Do not xfail: rerun against root's rebuilt artifact to qualify central mkdir
and implicit-parent participation. No Python stubs replace either producer.
"""
import pytest

from openviking.storage.errors import LockAcquisitionError, ResourceBusyError
from tests.storage.test_tree_replace_structured import binding_fs, indexed_fs, enabled_tree


@pytest.mark.parametrize("producer", ["mkdir", "ensure_parents_fast", "ensure_parents_multilevel"])
async def test_native_directory_producer_refuses_foreign_resources_tree(enabled_tree, producer):
    fs, _, ctx, target, *_ = enabled_tree
    agfs = fs._async_agfs
    root = fs._uri_to_path("viking://resources", ctx=ctx)
    lease = await agfs.pathlock_acquire_tree(root)
    child = root + "/published/native-contender"
    try:
        with pytest.raises((LockAcquisitionError, ResourceBusyError)):
            if producer == "mkdir":
                await agfs.mkdir(child)
            elif producer == "ensure_parents_fast":
                await agfs.ensure_parent_dirs(child + "/file.md")
            else:
                await agfs.ensure_parent_dirs(child + "/nested/deeper/file.md")
        assert not await fs.exists("viking://resources/published/native-contender", ctx=ctx)
    finally:
        await agfs.pathlock_release(lease)
