//! PathLockWrappedFS — automatic lock acquisition for simple file operations.
//!
//! This wrapper sits between `StatsWrappedFS` and `MountableFS` in the stack.
//! It auto-acquires locks for `create`, `mkdir`, `ensure_parent_dirs`, `write`,
//! `truncate`, `remove`, `remove_all`, `rename`, and `replace`, then delegates to the inner filesystem.
//!
//! It is NOT a lock protocol implementation — all lock logic is delegated to
//! `PathLockManager`.

use async_trait::async_trait;
use std::sync::Arc;
use tracing::debug;

use crate::core::filesystem::FileSystem;
use crate::core::internal_names::is_hidden_runtime_lock_name;
use crate::core::types::{
    FileInfo, GlobPage, GrepOptions, GrepResult, ListSortBy, SortOrder, TreeEntry, WriteFlag,
};
use crate::core::MountableFS;

use super::manager::{AutoPathLockAction, PathLockManager};
use super::types::{PathLockKind, PathLockRequest};

/// A `FileSystem` wrapper that auto-acquires path locks for mutating operations.
pub struct PathLockWrappedFS {
    /// The lock manager (single source of truth).
    manager: Arc<PathLockManager>,
    /// The inner filesystem.
    inner: Arc<dyn FileSystem>,
}

impl PathLockWrappedFS {
    /// Create a new PathLockWrappedFS.
    pub fn new(manager: Arc<PathLockManager>, inner: Arc<dyn FileSystem>) -> Self {
        Self { manager, inner }
    }

    /// Return the inner filesystem.
    pub fn inner_fs(&self) -> &Arc<dyn FileSystem> {
        &self.inner
    }

    /// Return true when `path` does not exist.
    ///
    /// Deleting a missing path has nothing to protect, and acquiring a lock
    /// there would create the path (or its ancestors) just to hold lock
    /// metadata. Return not-found without delegating a delete: the target
    /// could be created by another writer after this check.
    async fn is_missing(&self, path: &str) -> bool {
        matches!(
            self.inner.stat(path).await,
            Err(crate::core::Error::NotFound(_))
        )
    }

    /// Return true for virtual control paths and lock metadata that must not be auto-locked.
    fn should_bypass_auto_lock(path: &str) -> bool {
        path == "/queue"
            || path.starts_with("/queue/")
            || path == "/serverinfo"
            || path.starts_with("/serverinfo/")
            || path
                .rsplit_once('/')
                .map(|(_, name)| is_hidden_runtime_lock_name(name))
                .unwrap_or_else(|| is_hidden_runtime_lock_name(path))
    }

    /// Check if auto-locking should be skipped for the current context.
    ///
    /// Returns `true` when auto-locking is disabled or an active lease covers the operation.
    async fn should_skip_auto_lock(
        &self,
        requests: &[PathLockRequest],
    ) -> crate::core::Result<bool> {
        match self.manager.resolve_auto_pathlock_action(requests).await {
            Ok(AutoPathLockAction::Disabled) => {
                debug!(requests = ?requests, "pathlock wrapper skipped auto-lock because context disabled it");
                Ok(true)
            }
            Ok(AutoPathLockAction::Covered(lease)) => {
                debug!(lease_ref = %lease.lease.lease_ref, requests = ?requests, "pathlock wrapper skipped auto-lock because active lease already covers request");
                Ok(true)
            }
            Ok(AutoPathLockAction::Acquire) => {
                debug!(requests = ?requests, "pathlock wrapper will auto-acquire lease for request");
                Ok(false)
            }
            Err(error) => Err(crate::core::Error::internal(format!(
                "lock lease error: {error}"
            ))),
        }
    }

    /// Merge an operation result with its lock-release result.
    fn merge_operation_and_release<T>(
        operation: crate::core::Result<T>,
        release: super::types::PathLockResult<()>,
    ) -> crate::core::Result<T> {
        match (operation, release) {
            (Err(error), _) => Err(error),
            (Ok(value), Ok(())) => Ok(value),
            (Ok(_), Err(error)) => Err(crate::core::Error::internal(format!(
                "lock release error: {error}"
            ))),
        }
    }

    /// Return whether the routed encryption wrapper owns locking for this write.
    async fn encryption_handles_pathlock(&self, path: &str) -> bool {
        let any = self.inner.as_ref() as &dyn std::any::Any;
        match any.downcast_ref::<MountableFS>() {
            Some(mountable) => mountable.encryption_handles_pathlock(path).await,
            None => false,
        }
    }
}

#[async_trait]
impl FileSystem for PathLockWrappedFS {
    async fn create(&self, path: &str) -> crate::core::Result<()> {
        if Self::should_bypass_auto_lock(path) || self.encryption_handles_pathlock(path).await {
            return self.inner.create(path).await;
        }
        let requests = [PathLockRequest {
            path: path.to_string(),
            kind: PathLockKind::Exact,
        }];
        if self.should_skip_auto_lock(&requests).await? {
            return self.inner.create(path).await;
        }
        let lease = self
            .manager
            .acquire_exact(path, self.manager.default_lock_timeout(), None)
            .await?;
        let result = self.inner.create(path).await;
        let release = self.manager.release(&lease).await;
        Self::merge_operation_and_release(result, release)
    }

    async fn mkdir(&self, path: &str, mode: u32) -> crate::core::Result<()> {
        // EncryptionWrappedFS forwards directory creation without locking.
        if Self::should_bypass_auto_lock(path) {
            return self.inner.mkdir(path, mode).await;
        }
        let requests = [PathLockRequest {
            path: path.to_string(),
            kind: PathLockKind::Exact,
        }];
        if self.should_skip_auto_lock(&requests).await? {
            return self.inner.mkdir(path, mode).await;
        }
        // Exact on a missing path uses a parent sidecar: it does not create
        // the requested directory, preserving exclusive mkdir semantics.
        let lease = self
            .manager
            .acquire_exact(path, self.manager.default_lock_timeout(), None)
            .await?;
        let result = self.inner.mkdir(path, mode).await;
        let release = self.manager.release(&lease).await;
        Self::merge_operation_and_release(result, release)
    }

    async fn remove(&self, path: &str) -> crate::core::Result<()> {
        if Self::should_bypass_auto_lock(path) {
            return self.inner.remove(path).await;
        }
        if self.is_missing(path).await {
            return Err(crate::core::Error::NotFound(path.to_string()));
        }
        let requests = [PathLockRequest {
            path: path.to_string(),
            kind: PathLockKind::Exact,
        }];
        if self.should_skip_auto_lock(&requests).await? {
            return self.inner.remove(path).await;
        }
        let lease = self
            .manager
            .acquire_exact(path, self.manager.default_lock_timeout(), None)
            .await?;
        let result = self.inner.remove(path).await;
        let release = self.manager.release(&lease).await;
        Self::merge_operation_and_release(result, release)
    }

    async fn remove_all(&self, path: &str) -> crate::core::Result<()> {
        if Self::should_bypass_auto_lock(path) {
            return self.inner.remove_all(path).await;
        }
        if self.is_missing(path).await {
            return Err(crate::core::Error::NotFound(path.to_string()));
        }
        let requests = [PathLockRequest {
            path: path.to_string(),
            kind: PathLockKind::Tree,
        }];
        if self.should_skip_auto_lock(&requests).await? {
            return self.inner.remove_all(path).await;
        }
        let lease = self
            .manager
            .acquire_tree(path, self.manager.default_lock_timeout(), None)
            .await?;
        let result = self.inner.remove_all(path).await;
        let release = self.manager.release(&lease).await;
        Self::merge_operation_and_release(result, release)
    }

    async fn read(&self, path: &str, offset: u64, size: u64) -> crate::core::Result<Vec<u8>> {
        self.inner.read(path, offset, size).await
    }

    async fn write(
        &self,
        path: &str,
        data: &[u8],
        offset: u64,
        flags: WriteFlag,
    ) -> crate::core::Result<u64> {
        if Self::should_bypass_auto_lock(path) || self.encryption_handles_pathlock(path).await {
            return self.inner.write(path, data, offset, flags).await;
        }
        let requests = [PathLockRequest {
            path: path.to_string(),
            kind: PathLockKind::Exact,
        }];
        if self.should_skip_auto_lock(&requests).await? {
            return self.inner.write(path, data, offset, flags).await;
        }
        let lease = self
            .manager
            .acquire_exact(path, self.manager.default_lock_timeout(), None)
            .await?;
        let result = self.inner.write(path, data, offset, flags).await;
        let release = self.manager.release(&lease).await;
        Self::merge_operation_and_release(result, release)
    }

    async fn read_dir(
        &self,
        path: &str,
        offset: Option<usize>,
        limit: Option<usize>,
        sort_by: Option<ListSortBy>,
        sort_order: Option<SortOrder>,
    ) -> crate::core::Result<Vec<FileInfo>> {
        self.inner
            .read_dir(path, offset, limit, sort_by, sort_order)
            .await
    }

    async fn stat(&self, path: &str) -> crate::core::Result<FileInfo> {
        self.inner.stat(path).await
    }

    async fn rename(&self, old_path: &str, new_path: &str) -> crate::core::Result<()> {
        if Self::should_bypass_auto_lock(old_path) || Self::should_bypass_auto_lock(new_path) {
            return self.inner.rename(old_path, new_path).await;
        }

        // Determine if source is a directory.
        let src_is_dir = self
            .inner
            .stat(old_path)
            .await
            .map(|info| info.is_dir)
            .unwrap_or(false);

        let requests: Vec<PathLockRequest> = if src_is_dir {
            vec![
                PathLockRequest {
                    path: old_path.to_string(),
                    kind: PathLockKind::Tree,
                },
                PathLockRequest {
                    path: new_path.to_string(),
                    kind: PathLockKind::Exact,
                },
            ]
        } else {
            vec![
                PathLockRequest {
                    path: old_path.to_string(),
                    kind: PathLockKind::Exact,
                },
                PathLockRequest {
                    path: new_path.to_string(),
                    kind: PathLockKind::Exact,
                },
            ]
        };

        if self.should_skip_auto_lock(&requests).await? {
            return self.inner.rename(old_path, new_path).await;
        }

        let lease = self
            .manager
            .acquire_batch(&requests, self.manager.default_lock_timeout(), None)
            .await?;
        let result = self.inner.rename(old_path, new_path).await;
        let release = self.manager.release(&lease).await;
        Self::merge_operation_and_release(result, release)
    }

    async fn replace(&self, src_path: &str, dst_path: &str) -> crate::core::Result<()> {
        if Self::should_bypass_auto_lock(src_path) || Self::should_bypass_auto_lock(dst_path) {
            return self.inner.replace(src_path, dst_path).await;
        }

        let requests = vec![
            PathLockRequest {
                path: src_path.to_string(),
                kind: PathLockKind::Exact,
            },
            PathLockRequest {
                path: dst_path.to_string(),
                kind: PathLockKind::Exact,
            },
        ];
        if self.should_skip_auto_lock(&requests).await? {
            return self.inner.replace(src_path, dst_path).await;
        }
        let lease = self
            .manager
            .acquire_batch(&requests, self.manager.default_lock_timeout(), None)
            .await?;
        let result = self.inner.replace(src_path, dst_path).await;
        let release = self.manager.release(&lease).await;
        Self::merge_operation_and_release(result, release)
    }

    async fn chmod(&self, path: &str, mode: u32) -> crate::core::Result<()> {
        self.inner.chmod(path, mode).await
    }

    async fn truncate(&self, path: &str, size: u64) -> crate::core::Result<()> {
        if Self::should_bypass_auto_lock(path) || self.encryption_handles_pathlock(path).await {
            return self.inner.truncate(path, size).await;
        }
        let requests = [PathLockRequest {
            path: path.to_string(),
            kind: PathLockKind::Exact,
        }];
        if self.should_skip_auto_lock(&requests).await? {
            return self.inner.truncate(path, size).await;
        }
        let lease = self
            .manager
            .acquire_exact(path, self.manager.default_lock_timeout(), None)
            .await?;
        let result = self.inner.truncate(path, size).await;
        let release = self.manager.release(&lease).await;
        Self::merge_operation_and_release(result, release)
    }

    async fn grep(
        &self,
        path: &str,
        pattern: &str,
        options: GrepOptions<'_>,
    ) -> crate::core::Result<GrepResult> {
        self.inner.grep(path, pattern, options).await
    }

    async fn tree_directory(
        &self,
        path: &str,
        show_hidden: bool,
        node_limit: Option<usize>,
        level_limit: Option<usize>,
        offset: Option<usize>,
        sort_by: Option<ListSortBy>,
        sort_order: Option<SortOrder>,
        directories_only: bool,
    ) -> crate::core::Result<Vec<TreeEntry>> {
        self.inner
            .tree_directory(
                path,
                show_hidden,
                node_limit,
                level_limit,
                offset,
                sort_by,
                sort_order,
                directories_only,
            )
            .await
    }

    async fn glob_directory(
        &self,
        path: &str,
        pattern: &str,
        show_hidden: bool,
        page_size: Option<usize>,
        level_limit: Option<usize>,
        continuation_token: Option<String>,
    ) -> crate::core::Result<GlobPage> {
        self.inner
            .glob_directory(
                path,
                pattern,
                show_hidden,
                page_size,
                level_limit,
                continuation_token,
            )
            .await
    }

    async fn ensure_parent_dirs(&self, path: &str, mode: u32) -> crate::core::Result<()> {
        if Self::should_bypass_auto_lock(path) {
            return self.inner.ensure_parent_dirs(path, mode).await;
        }
        if path == "/" || !path.contains('/') {
            return self.inner.ensure_parent_dirs(path, mode).await;
        }
        let trimmed = path.trim_end_matches('/');
        let parent = match trimmed.rsplit_once('/') {
            Some(("", _)) | None => "/",
            Some((parent, _)) => parent,
        };
        // No mutation is needed when the parent already exists. Return here;
        // delegating create_dir_all could recreate a parent removed after stat.
        if matches!(self.inner.stat(parent).await, Ok(info) if info.is_dir) {
            return Ok(());
        }
        let requests = [PathLockRequest {
            path: parent.to_string(),
            kind: PathLockKind::Exact,
        }];
        if self.should_skip_auto_lock(&requests).await? {
            return self.inner.ensure_parent_dirs(path, mode).await;
        }
        // The manager checks ancestor Tree tokens for this deepest directory
        // before creating lock parents through the inner filesystem.
        let lease = self
            .manager
            .acquire_exact(parent, self.manager.default_lock_timeout(), None)
            .await?;
        let result = self.inner.ensure_parent_dirs(path, mode).await;
        let release = self.manager.release(&lease).await;
        Self::merge_operation_and_release(result, release)
    }
}

#[cfg(test)]
mod tests {
    use super::super::manager::PathLockConfig;
    use super::*;
    use crate::core::Error;
    use crate::lock::provider::MemoryPathLockProvider;
    use crate::plugins::memfs::MemFileSystem;

    /// Configurable stat result; any subsequent mutation would be a bug.
    struct NoMutationFS {
        existing: Option<FileInfo>,
    }

    #[async_trait]
    impl FileSystem for NoMutationFS {
        async fn stat(&self, path: &str) -> crate::core::Result<FileInfo> {
            self.existing
                .clone()
                .ok_or_else(|| Error::NotFound(path.to_string()))
        }

        async fn remove(&self, _path: &str) -> crate::core::Result<()> {
            panic!("must not delete after stat returned NotFound")
        }

        async fn remove_all(&self, _path: &str) -> crate::core::Result<()> {
            panic!("must not delete recursively after stat returned NotFound")
        }

        async fn create(&self, _path: &str) -> crate::core::Result<()> {
            panic!("must not create lock metadata")
        }

        async fn mkdir(&self, _path: &str, _mode: u32) -> crate::core::Result<()> {
            panic!("must not create a lock directory")
        }

        async fn read(
            &self,
            _path: &str,
            _offset: u64,
            _size: u64,
        ) -> crate::core::Result<Vec<u8>> {
            unreachable!()
        }

        async fn write(
            &self,
            _path: &str,
            _data: &[u8],
            _offset: u64,
            _flags: WriteFlag,
        ) -> crate::core::Result<u64> {
            panic!("must not write lock metadata")
        }

        async fn read_dir(
            &self,
            _path: &str,
            _offset: Option<usize>,
            _limit: Option<usize>,
            _sort_by: Option<ListSortBy>,
            _sort_order: Option<SortOrder>,
        ) -> crate::core::Result<Vec<FileInfo>> {
            unreachable!()
        }

        async fn rename(&self, _old_path: &str, _new_path: &str) -> crate::core::Result<()> {
            unreachable!()
        }

        async fn chmod(&self, _path: &str, _mode: u32) -> crate::core::Result<()> {
            unreachable!()
        }
    }

    #[tokio::test]
    async fn missing_deletes_return_without_delegating() {
        let fs: Arc<dyn FileSystem> = Arc::new(NoMutationFS { existing: None });
        let manager = Arc::new(PathLockManager::new(
            fs.clone(),
            Arc::new(MemoryPathLockProvider::new()),
            PathLockConfig::default(),
        ));
        let wrapped = PathLockWrappedFS::new(manager, fs);
        for recursive in [false, true] {
            let result = if recursive {
                wrapped.remove_all("/data/gone").await
            } else {
                wrapped.remove("/data/gone").await
            };
            assert!(matches!(result, Err(Error::NotFound(path)) if path == "/data/gone"));
        }
    }

    async fn wrapped() -> (PathLockWrappedFS, Arc<MemFileSystem>) {
        let fs = Arc::new(MemFileSystem::new());
        fs.mkdir("/data", 0o755).await.unwrap();
        let manager = Arc::new(PathLockManager::new(
            fs.clone() as Arc<dyn FileSystem>,
            Arc::new(MemoryPathLockProvider::new()),
            PathLockConfig::default(),
        ));
        (
            PathLockWrappedFS::new(manager, fs.clone() as Arc<dyn FileSystem>),
            fs,
        )
    }

    #[tokio::test]
    async fn mkdir_refuses_foreign_ancestor_tree_without_creating_target() {
        let (wrapped, fs) = wrapped().await;
        let foreign = wrapped
            .manager
            .acquire_tree("/data", std::time::Duration::ZERO, None)
            .await
            .unwrap();
        assert!(wrapped.mkdir("/data/new", 0o755).await.is_err());
        assert!(matches!(
            fs.stat("/data/new").await,
            Err(Error::NotFound(_))
        ));
        wrapped.manager.release(&foreign).await.unwrap();
    }

    #[tokio::test]
    async fn directory_creation_uses_own_covering_lease() {
        use crate::core::context::{FsContextInner, PathLockContext, FS_CTX};

        let (wrapped, fs) = wrapped().await;
        let lease = wrapped
            .manager
            .acquire_tree("/data", std::time::Duration::ZERO, None)
            .await
            .unwrap();
        let ctx = Arc::new(FsContextInner::with_pathlock(
            "tenant",
            PathLockContext {
                lease_ref: Some(lease.lease.lease_ref.clone()),
                disable_auto_pathlock: false,
            },
        ));
        FS_CTX
            .scope(ctx, async {
                wrapped.mkdir("/data/new", 0o755).await.unwrap();
                wrapped
                    .ensure_parent_dirs("/data/a/b/file", 0o755)
                    .await
                    .unwrap();
            })
            .await;
        assert!(fs.stat("/data/new").await.unwrap().is_dir);
        assert!(fs.stat("/data/a/b").await.unwrap().is_dir);
        wrapped.manager.release(&lease).await.unwrap();
    }

    #[tokio::test]
    async fn ensure_parents_fast_path_never_delegates_directory_creation() {
        let (_, memfs) = wrapped().await;
        let fs: Arc<dyn FileSystem> = Arc::new(NoMutationFS {
            existing: Some(memfs.stat("/data").await.unwrap()),
        });
        let manager = Arc::new(PathLockManager::new(
            fs.clone(),
            Arc::new(MemoryPathLockProvider::new()),
            PathLockConfig::default(),
        ));
        let wrapped = PathLockWrappedFS::new(manager, fs);
        wrapped
            .ensure_parent_dirs("/data/file", 0o755)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn ensure_parents_existing_fast_path_needs_no_lease() {
        let (wrapped, _) = wrapped().await;
        let foreign = wrapped
            .manager
            .acquire_tree("/data", std::time::Duration::ZERO, None)
            .await
            .unwrap();
        wrapped
            .ensure_parent_dirs("/data/file", 0o755)
            .await
            .unwrap();
        wrapped.manager.release(&foreign).await.unwrap();
    }

    #[tokio::test]
    async fn ensure_multilevel_parents_refuses_foreign_ancestor_tree() {
        let (wrapped, fs) = wrapped().await;
        let foreign = wrapped
            .manager
            .acquire_tree("/data", std::time::Duration::ZERO, None)
            .await
            .unwrap();
        assert!(wrapped
            .ensure_parent_dirs("/data/a/b/file", 0o755)
            .await
            .is_err());
        assert!(matches!(fs.stat("/data/a").await, Err(Error::NotFound(_))));
        wrapped.manager.release(&foreign).await.unwrap();
        wrapped
            .ensure_parent_dirs("/data/a/b/file", 0o755)
            .await
            .unwrap();
        assert!(fs.stat("/data/a/b").await.unwrap().is_dir);
    }

    #[tokio::test]
    async fn mkdir_keeps_exclusive_creation() {
        let (wrapped, _) = wrapped().await;
        wrapped.mkdir("/data/new", 0o755).await.unwrap();
        assert!(matches!(
            wrapped.mkdir("/data/new", 0o755).await,
            Err(Error::AlreadyExists(_))
        ));
    }

    #[tokio::test]
    async fn directory_control_paths_bypass_auto_lock() {
        let (wrapped, fs) = wrapped().await;
        fs.mkdir("/queue", 0o755).await.unwrap();
        let foreign = wrapped
            .manager
            .acquire_tree("/queue", std::time::Duration::ZERO, None)
            .await
            .unwrap();
        wrapped.mkdir("/queue/new", 0o755).await.unwrap();
        wrapped
            .ensure_parent_dirs("/queue/a/b/file", 0o755)
            .await
            .unwrap();
        wrapped.manager.release(&foreign).await.unwrap();
        let foreign = wrapped
            .manager
            .acquire_tree("/data", std::time::Duration::ZERO, None)
            .await
            .unwrap();
        wrapped.mkdir("/data/.path.ovlock", 0o755).await.unwrap();
        wrapped.manager.release(&foreign).await.unwrap();
    }

    #[tokio::test]
    async fn remove_missing_path_does_not_create_ancestors() {
        let (wrapped, fs) = wrapped().await;
        assert!(matches!(
            wrapped.remove("/data/gone/file.md").await,
            Err(Error::NotFound(_))
        ));
        assert!(matches!(
            fs.stat("/data/gone").await,
            Err(Error::NotFound(_))
        ));
    }

    #[tokio::test]
    async fn remove_all_missing_path_does_not_create_ancestors() {
        let (wrapped, fs) = wrapped().await;
        assert!(matches!(
            wrapped.remove_all("/data/gone/sub").await,
            Err(Error::NotFound(_))
        ));
        assert!(matches!(
            fs.stat("/data/gone").await,
            Err(Error::NotFound(_))
        ));
    }

    #[tokio::test]
    async fn remove_existing_path_still_locks_and_deletes() {
        let (wrapped, fs) = wrapped().await;
        fs.create("/data/a.md").await.unwrap();
        wrapped.remove("/data/a.md").await.unwrap();
        assert!(matches!(
            fs.stat("/data/a.md").await,
            Err(Error::NotFound(_))
        ));
    }
}
