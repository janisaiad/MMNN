# CPU/GPU eigensolver metadata recovery

The `K=16, M0=16, seed=0` training process completed on the H100 and wrote its
normal `COMPLETE` marker, but the original launcher revision rejected the
bundle while recomputing one diagnostic on CPU.

- Immutable training revision: `0df643907c591e2aeca48e9fff46574b7b2ae3fa`
- Manifest SHA-256: `96fdbb1dc364bbcc60ab68b9db2b8b982cef6771a89d9519ca00f3c9ae88dfea`
- Original failure-marker SHA-256: `99769daf2a3be33754295cf5d8642194f27dd89fc73725928dd81ac5a0226959`
- Task-bank SHA-256: `211faa1333de76518f8c216e7842f3526029988a18fb671df9b061a676cd7c4e`
- Run-record SHA-256: `6fc91305b34b379bd00534baea09b95d5c0a3f40a7c9f7d041a95cfd9143dd0c`
- GPU `task_covariance_min`: `0.0001773623371263966`
- CPU `task_covariance_min`: `0.00017724488861858845`

The regenerated CPU task bank was tensor-equal to the saved task bank. The
failure was therefore a backend difference in float32 `eigvalsh`, not task or
training drift. Commit `2998ec3` changes only the operational verifier: exact
task-tensor equality remains mandatory, while derived eigenspectrum metadata
uses a dtype-, dimension-, and spectral-scale-aware backward-error tolerance.
The patched verifier authenticated all ten files in the completed run before
the campaign was resumed; no checkpoint or scientific run artifact was edited.
