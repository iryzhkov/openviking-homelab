# Fixed backend candidate assembly

This recipe assembles files only. It does not start a container, server, CLI,
native module or queue consumer. Run it on agent-a through Chaldea, with
TMPDIR asserted as the job's physical disk-backed .cache/candidate directory,
and HOME asserted as $TMPDIR/home before writes.
Do not use the upstream Dockerfile or resolve dependencies.

Inputs are fixed in inputs.json. This locally prepared candidate pins product
a65f2fde663a51ed6b58601d251c51d7cd033006, tree
badba2e2cdd1bd4e37d15579436e64bc838d57f0. Local integration e00502df926b4c6c3a2cc3faa8b196bc1aeb3737,
tree 9996e4312da5f219191d62aee73424973113663c, preserves public main
0cde16d02fde0459a818ea69e01fcc1f4d2f4c94 and the candidate ancestry.
Only openviking/service/document_write.py changes in the selected product;
both dependency locks and all native Git inputs remain pinned. Root owns
final acceptance, source checks and assembly dispatch. No new candidate image
or digest exists from this preparation. The 915 tracked regular files under
openviking/ and openviking_cli/ are selected by local Git metadata. The supplied
selection.json records each path, Git mode, blob, SHA256 and size. Its digest
is pinned here. All tracked source LICENSE/NOTICE files are carried verbatim,
with their original paths under /usr/share/licenses/openviking-candidate for
notices outside the product packages. Product notices stay in their package.
This is file custody, not a legal compliance opinion.

The job requires the existing selected-source snapshot, the nine retained
native chunks, native-input.json, selection.json, job-binding.json and these
three recipe files. No chunks or native binary belong in public source.
job-binding.json records the staged candidate Git tree, parent head/tree,
recipe Git blobs/SHA256, exact native Git inputs and the Git-backed observation
that pyproject.toml and uv.lock have not changed since upstream 7f3ae1af.
The job rechecks every selected file and every chunk before assembly.
Root retains the native source-build receipt; matching source trees and the
retained binary digest establish correspondence, not a fresh reproducible build.

Typical job command, from the prepared snapshot:

    chaldea exec --host agent-a --queue-timeout 1m --artifact candidate-patch.tar -- timeout -k 15 600 sh -c 'jobroot=$(pwd -P); export TMPDIR="$jobroot/.cache/candidate"; export HOME="$TMPDIR/home"; test "$TMPDIR" = "$jobroot/.cache/candidate" && test "$HOME" = "$TMPDIR/home" && test "$(realpath -m "$TMPDIR")" = "$jobroot/.cache/candidate" && test "$(realpath -m "$HOME")" = "$jobroot/.cache/candidate/home" && test "$(stat -f -c %T "$jobroot")" != tmpfs && mkdir -p "$HOME" "$TMPDIR" && python3 -I source-ci/candidate/assemble.py'

The caller must assert these scratch paths before creating them, as the
qualification launcher does. The recipe repeats those assertions. Never run
assembly or syntax checks locally. It uses only the official hash/size-verified
go-containerregistry v0.22.1 archive and extracts only crane into scratch.
Actual version, executable hash, help and command arguments are retained.
No socket, daemon, privilege change or namespace workaround is involved.

The overlay is sorted by safe relative POSIX path, contains regular files only,
uses Git-derived 0644/0755 modes and native 0755, uid/gid 0, empty owner names
and mtime 0. PAX handles long paths. There are no symlinks, directory entries,
whiteouts or opaque-directory markers. It is built twice and compared byte for
byte; every output entry is checked against the selected bytes.

crane mutate uses --append and an explicit --output, which saves locally.
Entrypoint, working directory, user, command, exposed ports and health settings
are preserved. OPENVIKING_WITH_BOT=0 deliberately disables the unrelated bot.
PYTHONPYCACHEPREFIX=/var/empty/openviking-candidate-pycache redirects Python's
cache lookup to a prefix verified absent in every base layer.
PYTHONDONTWRITEBYTECODE=1 also prevents new caches. Base product .pyc files
must occur only under __pycache__; sourceless legacy .pyc is refused.
These defaults address stale bytecode file selection. Actual interpreter
behavior and compatibility remain runtime acceptance work.

The base supplies dependencies, packaged CLI, four engine extensions, generated
_version.py, entrypoint, health helper and built assets. Exact retained critical
hashes are checked; additive layer membership proves all other base objects
stay unchanged except the selected files. Preserve generated version as an
observed base value. Candidate labels identify accepted product, tooling tree
and native bytes separately, with base digest/revision/version retained.
No real config, keys, accounts or model endpoint is included.

The local output is a Docker-save tar, not an OCI-layout archive. Reloading it
with crane normalizes OCI descriptor media types to Docker schema2/config/layer
types. The recipe records both sides, verifies every compressed base layer byte,
and binds the exact reconstructed Docker manifest to crane digest --tarball.
It does not claim the pre-save OCI manifest digest. Config changes are bounded
to environment/identity labels, one rootfs diff ID and one layer history entry.

candidate-patch.tar is one regular artifact, capped at 64 MiB. It retains the
new compressed layer, exact candidate config and Docker manifest, public base
manifest/config/descriptors, Docker-save metadata and evidence inventories.
The unchanged 333,876,575 compressed base bytes are externally retrievable by
digest from ghcr.io/volcengine/openviking and are not returned. This patch bundle
is not a fully retained OCI archive. Durable custody still depends on retaining
the bundle and future base availability.

For exact reconstruction, verify custody.json, retrieve and hash the listed
base compressed blobs, and create a Docker-save tar with the recorded
docker-save-manifest.json names, retained candidate config and retained new
compressed layer. No source/native/config rebuild or recompression is needed.
The job performs this reconstruction from retained new objects plus the original
verified base blobs, then requires crane validate and digest equality.
The exact candidate distribution manifest is also retained by content digest.

File validation proves source/native placement and image/config/layer closure.
Runtime imports, packaged CLI execution, health/readiness, one serving process,
nonroot behavior, signals/shutdown and synthetic backend/HTTP contract tests
remain pending on an authorized OCI runtime. The prior job had cap0 and its
nested namespace probe failed. No runtime acceptance follows from this recipe.
Root owns independent review, push/PR, binary/registry release and deployment.
Live identity/config/mount/sole-writer evidence, protected isolated restore proof,
account creation-ID transition and queue capture/classification remain separate.
Handled failed deliveries can be ACKed; startup starts consumers before auth
identity initialization. Never infer safe live backfill from this image.
