# Account-bound private document embedding

The private document writer uses the account-bound async embedder selected by the actual RequestContext. Normal service initialization leaves the legacy query_embedder field unset, so dereferencing its synchronous embed method was invalid. The one production change calls the existing _get_embedder(ctx).embed_async API. Ownership/lifetime checks before and after the await, exact text, byte limit, empty-vector refusal and record assignment are preserved.

The faithful fresh-ADMIN/deferred-RAW test no longer installs a legacy field. Its red snapshot at baf3400409d720da65a35afc2edd6981d01b9eee failed at that exact AttributeError lookup after private preparation. This confirms the offline source mechanism, not the class or cause of a historical image failure. It uses the existing fake model factory through the normal account provider, original native identity/lease/store and real owning API/adapter.

The selected checks also cover provider absence with no publication, native lease loss during the awaited account embedding with no publication, and the existing document retention/control/backup/rollback cases. The two old embedding fault wrappers now patch the supported asynchronous AccountBoundEmbedder method. Existing fixtures/assertions remain, with only local fixture import aliases and legacy-field removal for this test module.

This branch starts from plain5ba1 and contains no diagnostic hook/overlay/logger. It does not alter initializer/provider/auth/API/schema/queue semantics. No image was built or run. Root owns source review, publication, image assembly and eventual separate normal-image qualification.
