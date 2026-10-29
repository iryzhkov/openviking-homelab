# Opt-in account embedding proof

These validation scripts follow CONTRIBUTING.md and are outside pytest's default testpaths=["tests"]; raw_write_proof.py also does not match python_files=["test_*.py"]. They are not portable product tests. The 12 ordinary cases remain in tests/server/test_document_write.py and require no PROOF variables or external adapter.

Run only in the existing explicitly provisioned isolated native environment:
    python -u "$PROOF_SOURCE/test_scripts/account_embedding/run_proof.py"

The working directory must contain the exact product tests, pyproject.toml and these test_scripts bytes. PROOF_SOURCE names the regular source snapshot; PROOF_OUT names a fresh public evidence directory. HOME must equal TMPDIR/home, with HOME/TMPDIR asserted below disk scratch before writes. Keep all private keys/data and raw pytest output under that HOME. The existing runner enforces bounded subprocess execution and exports only closed results and aggregate JUnit.

Prerequisites are native ragfs SHA8be7869a9f326ccd6cb67e093216f74296d1e98e93a73290918e6f33744ea1ad, the original pinned base dependencies and locked test dependencies. Supply the exact external HTTP adapter1c80c90e02a50a168c7f7ef02fd7c9aebaeb8319 under PROOF_SOURCE/http-input/ov_http and its requirements-http.lock dependencies. This external adapter is a validation input, not a product dependency. The proof asserts the actual imported document writer hash. The existing source-bound origin_plugin must be importable when using this evidence runner.

The explicit pytest target is:
    test_scripts/account_embedding/raw_write_proof.py::test_fresh_admin_deferred_raw_write

It runs three unchanged cases: fresh nonROOT ADMIN/deferred RAW publication with legacy field None and actual account-provider binding, missing-provider refusal, and native lease loss during awaited embedding. The success case requires committed+accepted before a bounded owning wait. Negative cases require zero formal publication and no mutation wait. Missing prerequisites fail this opt-in proof; they are not skips or passes. Ordinary pytest never needs to supply this environment.

Prior faithful RED and native8be evidence is retained separately. Relocation does not establish historical image causality or ordinary-image acceptance.
