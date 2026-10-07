# Archived interrupted resume

This directory preserves the complete partial state of
`k16_m0_8_query_attention_r16_m64_spread_seed1` at atomic checkpoint step 6500,
together with the worker failure publication from 2026-09-07.

No training update occurred during the failed resume. PyTorch loaded every
checkpoint tensor with `map_location=cuda`; consequently the archived CUDA RNG
state became a CUDA `uint8` tensor, while `torch.cuda.set_rng_state_all`
requires a CPU byte tensor. Continuing after deleting or ignoring that state
would not be an exact stochastic continuation, so the canonical campaign
restarts seed 1 from step zero under the unchanged source commit
`0df643907c591e2aeca48e9fff46574b7b2ae3fa`.

The completed seed 0 and the campaign manifest were not changed.
