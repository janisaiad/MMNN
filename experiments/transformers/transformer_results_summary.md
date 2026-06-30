# Transformer results summary

Source folder: `/Data/janis.aiad/MMNN/experiments/transformers`

Requested folder note: I did not find a folder named `ctransformers` under `/Data/janis.aiad/MMNN`, so this file is written in the existing `experiments/transformers` folder.

Recent history showed:

- `tail -f /Data/janis.aiad/MMNN/experiments/transformers/runs_transformers_all_20260522_020221.log`
- A queued scale run targeting `experiments/transformers/runs_dh_vs_K_scale_20260522_011654`, but no output folder or result files were found for that run.

Main run:

- Log file: `runs_transformers_all_20260522_020221.log`
- Started visible sequence: `Fri May 22 02:02:41 AM CEST 2026`
- Finished: `Fri May 22 02:46:29 AM CEST 2026`
- Result rows collected: `59`
- Experiment groups: `6`

Best result by experiment:

| Experiment | Best test_mse | Configuration |
| --- | ---: | --- |
| `runs_dh_vs_K` | 0.00416629 | `K=4`, `d_h=4`, `N=8192`, `Ttr=256`, `Tte=256`, `mixing=identity` |
| `runs_mixed_witnesses` | 0.0142508 | `K=4`, `d_h=4`, `N=16384`, `Ttr=256`, `Tte=128`, `mixing=random_wellcond` |
| `runs_train_prompt_scaling` | 0.0201894 | `K=8`, `d_h=8`, `N=32768`, `Ttr=128`, `Tte=256`, `mixing=identity` |
| `runs_N_scaling` | 0.0239510 | `K=8`, `d_h=4`, `N=32768`, `Ttr=256`, `Tte=256`, `mixing=identity` |
| `runs_test_prompt_scaling` | 0.0408149 | `K=8`, `d_h=8`, `N=16384`, `Ttr=256`, `Tte=128`, `mixing=identity` |
| `runs_slot_A` | 0.0510365 | `K=8`, `d_h=8`, `N=4096`, `Ttr=128`, `Tte=128`, `mixing=identity` |

All final results:

| Experiment | K | d_h | N | Ttr | Tte | Mixing | Step | test_mse | beta_mse | R_correct_mass | margin_mean |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `runs_N_scaling` | 8 | 4 | 128 | 256 | 256 | identity | 500 | 0.0916090 | 0.0916696 | 0.116485 | -0.0918364 |
| `runs_N_scaling` | 8 | 4 | 512 | 256 | 256 | identity | 2500 | 0.0521961 | 0.0526845 | 0.103366 | -0.200453 |
| `runs_N_scaling` | 8 | 4 | 2048 | 256 | 256 | identity | 2500 | 0.0310597 | 0.0310714 | 0.101813 | -0.215662 |
| `runs_N_scaling` | 8 | 4 | 8192 | 256 | 256 | identity | 2500 | 0.0293278 | 0.0287359 | 0.102017 | -0.227629 |
| `runs_N_scaling` | 8 | 4 | 32768 | 256 | 256 | identity | 2500 | 0.0239510 | 0.0240285 | 0.0975600 | -0.237211 |
| `runs_N_scaling` | 8 | 8 | 128 | 256 | 256 | identity | 2500 | 0.0943976 | 0.0929848 | 0.123322 | -0.0716239 |
| `runs_N_scaling` | 8 | 8 | 512 | 256 | 256 | identity | 2500 | 0.0503838 | 0.0506942 | 0.132104 | -0.0983441 |
| `runs_N_scaling` | 8 | 8 | 2048 | 256 | 256 | identity | 2500 | 0.0335025 | 0.0325287 | 0.144526 | -0.0940402 |
| `runs_N_scaling` | 8 | 8 | 8192 | 256 | 256 | identity | 2500 | 0.0268631 | 0.0259343 | 0.134861 | -0.141591 |
| `runs_N_scaling` | 8 | 8 | 32768 | 256 | 256 | identity | 2500 | 0.0249612 | 0.0245618 | 0.137350 | -0.140669 |
| `runs_N_scaling` | 8 | 16 | 128 | 256 | 256 | identity | 1500 | 0.0933626 | 0.0923648 | 0.130186 | -0.0155158 |
| `runs_N_scaling` | 8 | 16 | 512 | 256 | 256 | identity | 2500 | 0.0567021 | 0.0568050 | 0.140875 | -0.0376634 |
| `runs_N_scaling` | 8 | 16 | 2048 | 256 | 256 | identity | 2500 | 0.0361501 | 0.0354421 | 0.141382 | -0.0683085 |
| `runs_N_scaling` | 8 | 16 | 8192 | 256 | 256 | identity | 2500 | 0.0321364 | 0.0308495 | 0.149484 | -0.0855627 |
| `runs_N_scaling` | 8 | 16 | 32768 | 256 | 256 | identity | 2500 | 0.0261944 | 0.0264847 | 0.153329 | -0.0821199 |
| `runs_dh_vs_K` | 4 | 2 | 8192 | 256 | 256 | identity | 2000 | 0.00627535 | 0.00630662 | 0.248243 | -0.0369769 |
| `runs_dh_vs_K` | 4 | 4 | 8192 | 256 | 256 | identity | 2000 | 0.00416629 | 0.00423544 | 0.180918 | -0.261931 |
| `runs_dh_vs_K` | 4 | 8 | 8192 | 256 | 256 | identity | 2000 | 0.00622856 | 0.00645077 | 0.302732 | 0.00204731 |
| `runs_dh_vs_K` | 4 | 16 | 8192 | 256 | 256 | identity | 2000 | 0.00624000 | 0.00627123 | 0.250655 | -0.0552581 |
| `runs_dh_vs_K` | 4 | 32 | 8192 | 256 | 256 | identity | 2000 | 0.00448893 | 0.00468029 | 0.299670 | -0.0449655 |
| `runs_dh_vs_K` | 4 | 64 | 8192 | 256 | 256 | identity | 2000 | 0.00632517 | 0.00649447 | 0.235205 | -0.0797203 |
| `runs_dh_vs_K` | 8 | 2 | 8192 | 256 | 256 | identity | 2000 | 0.0394882 | 0.0408159 | 0.132213 | -0.103628 |
| `runs_dh_vs_K` | 8 | 4 | 8192 | 256 | 256 | identity | 2000 | 0.0308358 | 0.0321163 | 0.104778 | -0.197147 |
| `runs_dh_vs_K` | 8 | 8 | 8192 | 256 | 256 | identity | 2000 | 0.0340517 | 0.0346393 | 0.133848 | -0.114889 |
| `runs_dh_vs_K` | 8 | 16 | 8192 | 256 | 256 | identity | 2000 | 0.0385917 | 0.0388785 | 0.137798 | -0.0724480 |
| `runs_dh_vs_K` | 8 | 32 | 8192 | 256 | 256 | identity | 2000 | 0.0362625 | 0.0370045 | 0.133125 | -0.0679022 |
| `runs_dh_vs_K` | 8 | 64 | 8192 | 256 | 256 | identity | 2000 | 0.0380630 | 0.0391889 | 0.141926 | -0.0632457 |
| `runs_dh_vs_K` | 16 | 2 | 8192 | 256 | 256 | identity | 2000 | 0.145553 | 0.148030 | 0.0673617 | -0.0685213 |
| `runs_dh_vs_K` | 16 | 4 | 8192 | 256 | 256 | identity | 2000 | 0.150671 | 0.153122 | 0.0636454 | -0.0707923 |
| `runs_dh_vs_K` | 16 | 8 | 8192 | 256 | 256 | identity | 2000 | 0.143576 | 0.146319 | 0.0623379 | -0.0818716 |
| `runs_dh_vs_K` | 16 | 16 | 8192 | 256 | 256 | identity | 2000 | 0.155788 | 0.159897 | 0.0637568 | -0.0434062 |
| `runs_dh_vs_K` | 16 | 32 | 8192 | 256 | 256 | identity | 2000 | 0.164335 | 0.166916 | 0.0619732 | -0.0494974 |
| `runs_dh_vs_K` | 16 | 64 | 8192 | 256 | 256 | identity | 2000 | 0.160424 | 0.162395 | 0.0624640 | -0.0494904 |
| `runs_dh_vs_K` | 32 | 2 | 8192 | 256 | 256 | identity | 2000 | 0.765024 | 0.784286 | 0.0314472 | -0.261383 |
| `runs_dh_vs_K` | 32 | 4 | 8192 | 256 | 256 | identity | 2000 | 0.513951 | 0.536079 | 0.0325537 | -0.153053 |
| `runs_dh_vs_K` | 32 | 8 | 8192 | 256 | 256 | identity | 2000 | 0.544643 | 0.566610 | 0.0310196 | -0.0898857 |
| `runs_dh_vs_K` | 32 | 16 | 8192 | 256 | 256 | identity | 2000 | 0.556916 | 0.573727 | 0.0317802 | -0.0547333 |
| `runs_dh_vs_K` | 32 | 32 | 8192 | 256 | 256 | identity | 2000 | 0.604744 | 0.620936 | 0.0307815 | -0.0577338 |
| `runs_dh_vs_K` | 32 | 64 | 8192 | 256 | 256 | identity | 2000 | 0.580872 | 0.593648 | 0.0311571 | -0.0505670 |
| `runs_mixed_witnesses` | 4 | 4 | 16384 | 256 | 128 | random_wellcond | 3000 | 0.0142508 | 0.0145992 | 0.181084 | -0.409366 |
| `runs_mixed_witnesses` | 4 | 8 | 16384 | 256 | 128 | random_wellcond | 3000 | 0.0173434 | 0.0183446 | 0.248010 | -0.179041 |
| `runs_mixed_witnesses` | 4 | 16 | 16384 | 256 | 128 | random_wellcond | 3000 | 0.0168128 | 0.0179527 | 0.236464 | -0.131738 |
| `runs_mixed_witnesses` | 4 | 32 | 16384 | 256 | 128 | random_wellcond | 3000 | 0.0162277 | 0.0164773 | 0.233419 | -0.173985 |
| `runs_mixed_witnesses` | 8 | 4 | 16384 | 256 | 128 | random_wellcond | 3000 | 0.0751423 | 0.0742374 | 0.123826 | -0.216567 |
| `runs_mixed_witnesses` | 8 | 8 | 16384 | 256 | 128 | random_wellcond | 3000 | 0.0850826 | 0.0835342 | 0.108094 | -0.232135 |
| `runs_mixed_witnesses` | 8 | 16 | 16384 | 256 | 128 | random_wellcond | 3000 | 0.0925012 | 0.0897094 | 0.149991 | -0.0869017 |
| `runs_mixed_witnesses` | 8 | 32 | 16384 | 256 | 128 | random_wellcond | 3000 | 0.0893979 | 0.0873832 | 0.120373 | -0.199880 |
| `runs_mixed_witnesses` | 16 | 4 | 16384 | 256 | 128 | random_wellcond | 3000 | 0.302860 | 0.309240 | 0.0651106 | -0.0875507 |
| `runs_mixed_witnesses` | 16 | 8 | 16384 | 256 | 128 | random_wellcond | 3000 | 0.307257 | 0.310983 | 0.0642760 | -0.101045 |
| `runs_mixed_witnesses` | 16 | 16 | 16384 | 256 | 128 | random_wellcond | 3000 | 0.325506 | 0.324454 | 0.0588359 | -0.108374 |
| `runs_mixed_witnesses` | 16 | 32 | 16384 | 256 | 128 | random_wellcond | 3000 | 0.315342 | 0.319224 | 0.0629694 | -0.0998412 |
| `runs_train_prompt_scaling` | 8 | 8 | 32768 | 8 | 256 | identity | 2500 | 0.0480247 | 0.0487000 | 0.139120 | -0.237609 |
| `runs_train_prompt_scaling` | 8 | 8 | 32768 | 16 | 256 | identity | 2500 | 0.0347949 | 0.0344879 | 0.135503 | -0.283850 |
| `runs_train_prompt_scaling` | 8 | 8 | 32768 | 32 | 256 | identity | 2000 | 0.0293879 | 0.0294818 | 0.141120 | -0.208989 |
| `runs_train_prompt_scaling` | 8 | 8 | 32768 | 64 | 256 | identity | 2500 | 0.0228453 | 0.0228749 | 0.135060 | -0.207480 |
| `runs_train_prompt_scaling` | 8 | 8 | 32768 | 128 | 256 | identity | 2500 | 0.0201894 | 0.0201014 | 0.134844 | -0.183710 |
| `runs_train_prompt_scaling` | 8 | 8 | 32768 | 256 | 256 | identity | 2500 | 0.0249612 | 0.0245618 | 0.137350 | -0.140669 |
| `runs_slot_A` | 8 | 8 | 4096 | 128 | 128 | identity | 2000 | 0.0510365 | 0.0511877 | 0.141473 | -0.130847 |
| `runs_test_prompt_scaling` | 8 | 8 | 16384 | 256 | 128 | identity | 3000 | 0.0408149 | 0.0401027 | 0.141416 | -0.134206 |
