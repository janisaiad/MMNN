import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # we locate repo root #
if str(_ROOT) not in sys.path:  # we ensure imports from repo root #
    sys.path.insert(0, str(_ROOT))  # we add path #

from experiments.helmholtz.data_gen import HelmholtzDatasetConfig, generate_dataset_npz
from experiments.helmholtz.dataset import HelmholtzIOConfig
from experiments.helmholtz.logging_utils import dump_json, now_ts
from experiments.helmholtz.train import ModelConfig, TrainConfig, run_one
from experiments.helmholtz.summarize import write_summary


def build_ablation_suite(io_cfg: HelmholtzIOConfig) -> list[tuple[str, ModelConfig]]:  # we define ablation configs #
    in_channels = 4 + (2 if io_cfg.use_coords else 0) + (1 if io_cfg.include_bmask else 0)  # we compute channels #

    suite: list[tuple[str, ModelConfig]] = []  # we collect #
    suite.append(
        (
            "fno_baseline",
            ModelConfig(
                model_type="fno",
                in_channels=in_channels,
                fno_width=64,
                fno_depth=4,
                fno_modes=12,
            ),
        )
    )  # we add fno #

    suite.append(
        (
            "deeponet_trunkMMNN_branchMLP",
            ModelConfig(
                model_type="deeponet",
                in_channels=in_channels,
                deeponet_latent_dim=128,
                trunk_depth=3,
                trunk_width=256,
                trunk_rank=64,
                trunk_fix_wb=True,
                trunk_normalize_output=False,
                branch_type="mlp",
                branch_hidden=512,
                branch_depth=4,
                sensors_grid=16,
            ),
        )
    )  # we add deeponet #

    suite.append(
        (
            "deeponet_trunkMMNN_branchMMNN_RF",
            ModelConfig(
                model_type="deeponet",
                in_channels=in_channels,
                deeponet_latent_dim=128,
                trunk_depth=3,
                trunk_width=256,
                trunk_rank=64,
                trunk_fix_wb=True,
                trunk_normalize_output=False,
                branch_type="mmnn",
                branch_mmnn_depth=4,
                branch_mmnn_width=512,
                branch_mmnn_rank=512,
                branch_fix_wb=True,
                sensors_grid=16,
            ),
        )
    )  # we add branch rf #

    suite.append(
        (
            "deeponet_trunkMMNN_branchMMNN_LR",
            ModelConfig(
                model_type="deeponet",
                in_channels=in_channels,
                deeponet_latent_dim=128,
                trunk_depth=3,
                trunk_width=256,
                trunk_rank=64,
                trunk_fix_wb=True,
                trunk_normalize_output=False,
                branch_type="mmnn",
                branch_mmnn_depth=4,
                branch_mmnn_width=1024,
                branch_mmnn_rank=32,
                branch_fix_wb=False,
                sensors_grid=16,
            ),
        )
    )  # we add branch low rank #

    suite.append(
        (
            "deeponet_trunkMMNN_branchMMNN_RFLR",
            ModelConfig(
                model_type="deeponet",
                in_channels=in_channels,
                deeponet_latent_dim=128,
                trunk_depth=3,
                trunk_width=256,
                trunk_rank=64,
                trunk_fix_wb=True,
                trunk_normalize_output=False,
                branch_type="mmnn",
                branch_mmnn_depth=4,
                branch_mmnn_width=1024,
                branch_mmnn_rank=32,
                branch_fix_wb=True,
                sensors_grid=16,
            ),
        )
    )  # we add branch rf-lr #

    return suite  # we return #


def main() -> None:  # we run pipeline #
    parser = argparse.ArgumentParser()  # we build parser #
    parser.add_argument("--n_grid", type=int, default=64)  # we set grid #
    parser.add_argument("--n_train", type=int, default=256)  # we set train #
    parser.add_argument("--n_test", type=int, default=64)  # we set test #
    parser.add_argument("--epochs", type=int, default=50)  # we set epochs #
    parser.add_argument("--batch", type=int, default=8)  # we set batch #
    parser.add_argument("--lr", type=float, default=1e-3)  # we set lr #
    parser.add_argument("--device", type=str, default="cuda")  # we set device #
    parser.add_argument("--seed", type=int, default=0)  # we set seed #
    parser.add_argument("--only", type=str, default="")  # we filter runs #
    args = parser.parse_args()  # we parse #

    base = Path("experiments/helmholtz")  # we set base #
    base.mkdir(parents=True, exist_ok=True)  # we ensure #

    dataset_cfg = HelmholtzDatasetConfig(n_grid=int(args.n_grid), n_train=int(args.n_train), n_test=int(args.n_test))  # we set dataset cfg #
    dataset_path = base / "data" / f"helmholtz_n{dataset_cfg.n_grid}_train{dataset_cfg.n_train}_test{dataset_cfg.n_test}.npz"  # we set path #
    if not dataset_path.exists():  # we generate #
        generate_dataset_npz(dataset_path, dataset_cfg)  # we generate #

    io_cfg = HelmholtzIOConfig(use_coords=True, include_bmask=False)  # we set io #
    suite = build_ablation_suite(io_cfg)  # we build suite #

    train_cfg = TrainConfig(
        seed=int(args.seed),
        device=str(args.device),
        batch_size=int(args.batch),
        num_epochs=int(args.epochs),
        lr=float(args.lr),
        log_every=10,
    )  # we set train #

    tag = f"bench_{now_ts()}"  # we tag #
    root = base / "runs" / tag  # we set root #
    root.mkdir(parents=True, exist_ok=True)  # we ensure #
    dump_json(root / "benchmark_config.json", {"dataset_path": str(dataset_path), "train_cfg": train_cfg.__dict__, "io_cfg": io_cfg.__dict__})  # we save #

    for name, model_cfg in suite:  # we iterate #
        if args.only and args.only not in name:  # we filter #
            continue  # we skip #
        run_dir = root / name  # we set #
        run_one(run_dir, dataset_path, dataset_cfg, io_cfg, model_cfg, train_cfg)  # we run #

    write_summary(base / "runs" / tag, base / "runs" / tag / "_summary")  # we summarize #


if __name__ == "__main__":  # we run #
    main()  # we run #

