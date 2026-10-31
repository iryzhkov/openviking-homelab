# Focused document snapshot qualification

`document-snapshots.yml` exports the checked-out source without Git metadata and runs `run_document_snapshot_checks.sh`. The script is the exact harness that passed the 20 document-write and 30 reindex effect-bounds cases in Chaldea at product commit `9a64acd7756f2611cef4cd35221e679cfabf115d`.

The harness pins four source/test Git blobs, verifies the anonymous dependency image and every layer, overlays and compares the current Python source and tests, and records native hashes and module origins. A future change to those four files needs an explicit pin update and fresh acceptance. The fixture uses the ordinary native local storage factory, synthetic model providers, and real queue/tracker functions. It omits only the existing mock-only SessionCommit fallback. Tests cannot connect to external services, and all writes use asserted disk scratch HOME/TMPDIR.

The 50 cases cover snapshot/backup/rollback, restricted outsider access, direct control-target and reindex refusal, canonical identity, and reindex effect boundaries. The current regression's baseline failed during `document_node` on `.relations.json`, before replacement/backup; the separate baseline job's overly strict post-validator failed too, so that job is not reported as passing.

The native libraries come from a digest-verified dependency image and were not rebuilt from this source revision. Test execution does not prove the held live OV note has a relations sidecar, that deployment resolves its refusal, or that the curator pilot or scheduling qualifies. The historical filter-only `source-checks.yml` attestation remains unchanged and is not applicable to this batch.
