import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # we locate repo root #
if str(_ROOT) not in sys.path:  # we ensure imports from repo root #
    sys.path.insert(0, str(_ROOT))  # we add path #

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from experiments.helmholtz.data_gen import HelmholtzDatasetConfig, generate_dataset_npz
from experiments.helmholtz.dataset import HelmholtzIOConfig, HelmholtzNPZDataset
from experiments.helmholtz.logging_utils import Tee, append_jsonl, dump_json, now_ts, set_all_seeds
from experiments.helmholtz.models import FNO2d, DeepONetGrid, build_branch_mlp, build_branch_mmnn, build_trunk_mmnn
from experiments.helmholtz.plots import plot_metrics_jsonl


@dataclass(frozen=True)
class TrainConfig:  # we store training hyperparameters #
    seed: int = 0  # we set seed #
    device: str = "cuda"  # we set device #
    batch_size: int = 8  # we set batch size #
    num_epochs: int = 50  # we set epochs #
    lr: float = 1e-3  # we set lr #
    weight_decay: float = 0.0  # we set wd #
    grad_clip: float = 0.0  # we set clip #
    log_every: int = 10  # we log often #


@dataclass(frozen=True)
class ModelConfig:  # we store model hyperparameters #
    model_type: str = "fno"  # we set fno or deeponet #
    in_channels: int = 6  # we set input channels for fno #
    fno_width: int = 64  # we set width #
    fno_depth: int = 4  # we set depth #
    fno_modes: int = 12  # we set modes #
    deeponet_latent_dim: int = 128  # we set latent dim #
    trunk_depth: int = 4  # we set trunk depth #
    trunk_width: int = 512  # we set trunk width #
    trunk_rank: int = 32  # we set trunk rank #
    trunk_fix_wb: bool = True  # we freeze trunk random features #
    trunk_normalize_output: bool = False  # we disable output normalization for trunk basis #
    branch_type: str = "mlp"  # we set branch family #
    branch_hidden: int = 512  # we set hidden dim #
    branch_depth: int = 4  # we set depth #
    branch_mmnn_depth: int = 4  # we set mmnn depth #
    branch_mmnn_width: int = 1024  # we set mmnn width #
    branch_mmnn_rank: int = 32  # we set mmnn rank #
    branch_fix_wb: bool = False  # we set branch fix_wb #
    sensors_grid: int = 16  # we set sensors grid (S=s^2) #


def _build_coords(n: int, device: torch.device) -> torch.Tensor:  # we build grid coords #
    xs = torch.linspace(-1.0, 1.0, n, device=device)  # we build #
    x, y = torch.meshgrid(xs, xs, indexing="ij")  # we meshgrid #
    coords = torch.stack([x, y], dim=-1)  # we stack #
    return coords  # we return (H,W,2) #


def _extract_sensors(x: torch.Tensor, s: int) -> torch.Tensor:  # we extract sensor values from input channels #
    b, c, h, w = x.shape  # we unpack #
    ss = int(s)  # we cast #
    ii = torch.linspace(0, h - 1, ss, device=x.device).round().long()  # we pick indices #
    jj = torch.linspace(0, w - 1, ss, device=x.device).round().long()  # we pick indices #
    grid_i, grid_j = torch.meshgrid(ii, jj, indexing="ij")  # we mesh #
    vals = x[:, :, grid_i, grid_j]  # we gather (B,C,ss,ss) #
    flat = vals.reshape(b, c * ss * ss)  # we flatten #
    return flat  # we return #


def build_model(model_cfg: ModelConfig, io_cfg: HelmholtzIOConfig, n_grid: int, device: torch.device) -> nn.Module:  # we build model #
    if model_cfg.model_type == "fno":  # we dispatch #
        return FNO2d(
            in_channels=int(model_cfg.in_channels),
            out_channels=1,
            width=int(model_cfg.fno_width),
            depth=int(model_cfg.fno_depth),
            modes1=int(model_cfg.fno_modes),
            modes2=int(model_cfg.fno_modes),
            padding=6,
        ).to(device)  # we return #

    if model_cfg.model_type != "deeponet":  # we validate #
        raise ValueError(f"unknown model_type={model_cfg.model_type!r}")  # we raise #

    coords = _build_coords(int(n_grid), device=device)  # we build coords #
    trunk = build_trunk_mmnn(
        latent_dim=int(model_cfg.deeponet_latent_dim),
        depth=int(model_cfg.trunk_depth),
        width=int(model_cfg.trunk_width),
        rank=int(model_cfg.trunk_rank),
        device=str(device),
        fix_wb=bool(model_cfg.trunk_fix_wb),
        resnet=False,
        normalize_output=bool(model_cfg.trunk_normalize_output),
    ).to(device)  # we build trunk #
    trunk_is_fixed = bool(model_cfg.trunk_fix_wb)  # we treat fix_wb as fixed basis #
    if trunk_is_fixed:  # we freeze full trunk when we use it as a basis #
        for p in trunk.parameters():  # we loop #
            p.requires_grad = False  # we freeze #

    in_dim_branch = int(model_cfg.in_channels) * int(model_cfg.sensors_grid) * int(model_cfg.sensors_grid)  # we set sensor dim #
    if model_cfg.branch_type == "mlp":  # we build mlp branch #
        branch = build_branch_mlp(
            in_dim=int(in_dim_branch),
            latent_dim=int(model_cfg.deeponet_latent_dim),
            hidden_dim=int(model_cfg.branch_hidden),
            depth=int(model_cfg.branch_depth),
            activation="relu",
        ).to(device)  # we build #
    elif model_cfg.branch_type == "mmnn":  # we build mmnn branch #
        branch = build_branch_mmnn(
            in_dim=int(in_dim_branch),
            latent_dim=int(model_cfg.deeponet_latent_dim),
            depth=int(model_cfg.branch_mmnn_depth),
            width=int(model_cfg.branch_mmnn_width),
            rank=int(model_cfg.branch_mmnn_rank),
            device=str(device),
            fix_wb=bool(model_cfg.branch_fix_wb),
            resnet=False,
        ).to(device)  # we build #
    else:  # we fail #
        raise ValueError(f"unknown branch_type={model_cfg.branch_type!r}")  # we raise #

    model = DeepONetGrid(
        branch=branch,
        trunk=trunk,
        coords_hw=coords,
        latent_dim=int(model_cfg.deeponet_latent_dim),
        trunk_is_fixed=trunk_is_fixed,
    ).to(device)  # we build deeponet grid #
    model.maybe_cache_trunk()  # we cache fixed trunk #
    return model  # we return #


def masked_relative_l2(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # we compute relative l2 on domain #
    num = torch.sum(((pred - target) * mask) ** 2, dim=(1, 2, 3))  # we sum #
    den = torch.sum((target * mask) ** 2, dim=(1, 2, 3)) + 1e-12  # we sum #
    return torch.sqrt(num / den)  # we return per-sample #


def run_one(
    run_dir: Path,
    dataset_path: Path,
    dataset_cfg: HelmholtzDatasetConfig,
    io_cfg: HelmholtzIOConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> Path:  # we run a single training job #
    run_dir = Path(run_dir)  # we normalize #
    run_dir.mkdir(parents=True, exist_ok=True)  # we ensure #
    dump_json(run_dir / "dataset_config.json", asdict(dataset_cfg))  # we save config #
    dump_json(run_dir / "io_config.json", asdict(io_cfg))  # we save config #
    dump_json(run_dir / "model_config.json", asdict(model_cfg))  # we save config #
    dump_json(run_dir / "train_config.json", asdict(train_cfg))  # we save config #

    tee = Tee(run_dir / "stdout.log")  # we tee #
    old_stdout, old_stderr = (torch.sys.stdout if hasattr(torch, "sys") else None), None  # we keep placeholder #
    import sys  # we import #
    sys.stdout = tee  # we redirect #
    sys.stderr = tee  # we redirect #

    try:  # we run #
        set_all_seeds(int(train_cfg.seed))  # we seed #
        device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")  # we set device #
        print(f"run_dir={run_dir}")  # we log #
        print(f"device={device}")  # we log #
        print(f"dataset_path={dataset_path}")  # we log #

        ds_train = HelmholtzNPZDataset(dataset_path, split="train", io_cfg=io_cfg)  # we load #
        ds_test = HelmholtzNPZDataset(dataset_path, split="test", io_cfg=io_cfg)  # we load #
        n_grid = int(ds_train.n_grid)  # we store #
        loader_train = DataLoader(ds_train, batch_size=int(train_cfg.batch_size), shuffle=True, num_workers=0)  # we build loader #
        loader_test = DataLoader(ds_test, batch_size=int(train_cfg.batch_size), shuffle=False, num_workers=0)  # we build loader #

        model = build_model(model_cfg=model_cfg, io_cfg=io_cfg, n_grid=n_grid, device=device)  # we build model #
        print(f"trainable_parameters={sum(p.numel() for p in model.parameters() if p.requires_grad)}")  # we log #

        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(train_cfg.lr), weight_decay=float(train_cfg.weight_decay))  # we build optimizer #
        loss_fn = nn.MSELoss(reduction="none")  # we use elementwise mse #

        writer = SummaryWriter(log_dir=str(run_dir / "tb"))  # we build tb #
        metrics_path = run_dir / "metrics.jsonl"  # we set metrics file #

        global_step = 0  # we init #
        for epoch in range(1, int(train_cfg.num_epochs) + 1):  # we loop #
            model.train()  # we train #
            losses = []  # we collect #
            rels = []  # we collect #
            for batch in loader_train:  # we iterate #
                x = batch["x"].to(device)  # we move #
                u = batch["u"].to(device)  # we move #
                mask = batch["mask"].to(device)  # we move #
                opt.zero_grad(set_to_none=True)  # we clear #
                if model_cfg.model_type == "fno":  # we forward #
                    pred = model(x)  # we predict #
                else:  # we deeponet #
                    sensors = _extract_sensors(x, s=int(model_cfg.sensors_grid))  # we extract #
                    pred = model(sensors, hw=(n_grid, n_grid))  # we predict #
                mse = loss_fn(pred, u)  # we compute #
                loss = torch.mean(mse * mask) / (torch.mean(mask) + 1e-12)  # we mask-normalize #
                loss.backward()  # we backprop #
                if float(train_cfg.grad_clip) > 0.0:  # we clip #
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(train_cfg.grad_clip))  # we clip #
                opt.step()  # we step #
                with torch.no_grad():  # we compute metrics #
                    rel = masked_relative_l2(pred, u, mask).mean()  # we compute mean #
                losses.append(float(loss.detach().cpu().item()))  # we store #
                rels.append(float(rel.detach().cpu().item()))  # we store #
                if global_step % int(train_cfg.log_every) == 0:  # we log #
                    writer.add_scalar("train/loss", losses[-1], global_step=global_step)  # we log #
                    writer.add_scalar("train/rel_l2", rels[-1], global_step=global_step)  # we log #
                global_step += 1  # we increment #

            train_loss = float(np.mean(losses))  # we aggregate #
            train_rel = float(np.mean(rels))  # we aggregate #

            model.eval()  # we eval #
            test_losses = []  # we collect #
            test_rels = []  # we collect #
            with torch.no_grad():  # we eval #
                for batch in loader_test:  # we iterate #
                    x = batch["x"].to(device)  # we move #
                    u = batch["u"].to(device)  # we move #
                    mask = batch["mask"].to(device)  # we move #
                    if model_cfg.model_type == "fno":  # we forward #
                        pred = model(x)  # we predict #
                    else:  # we forward #
                        sensors = _extract_sensors(x, s=int(model_cfg.sensors_grid))  # we extract #
                        pred = model(sensors, hw=(n_grid, n_grid))  # we predict #
                    mse = loss_fn(pred, u)  # we compute #
                    loss = torch.mean(mse * mask) / (torch.mean(mask) + 1e-12)  # we normalize #
                    rel = masked_relative_l2(pred, u, mask).mean()  # we compute #
                    test_losses.append(float(loss.cpu().item()))  # we store #
                    test_rels.append(float(rel.cpu().item()))  # we store #

            test_loss = float(np.mean(test_losses))  # we aggregate #
            test_rel = float(np.mean(test_rels))  # we aggregate #

            writer.add_scalar("epoch/train_loss", train_loss, global_step=epoch)  # we log #
            writer.add_scalar("epoch/test_loss", test_loss, global_step=epoch)  # we log #
            writer.add_scalar("epoch/train_rel_l2", train_rel, global_step=epoch)  # we log #
            writer.add_scalar("epoch/test_rel_l2", test_rel, global_step=epoch)  # we log #

            row = {
                "epoch": int(epoch),
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_rel_l2": train_rel,
                "test_rel_l2": test_rel,
            }  # we build row #
            append_jsonl(metrics_path, row)  # we append #
            print(f"epoch={epoch} train_loss={train_loss:.3e} test_loss={test_loss:.3e} train_rel={train_rel:.3e} test_rel={test_rel:.3e}")  # we print #

            if epoch == int(train_cfg.num_epochs) or epoch % max(1, int(train_cfg.num_epochs) // 5) == 0:  # we checkpoint #
                torch.save({"model": model.state_dict(), "epoch": epoch, "model_cfg": asdict(model_cfg)}, run_dir / f"ckpt_epoch{epoch}.pt")  # we save #

        writer.flush()  # we flush #
        writer.close()  # we close #
        try:  # we try plotting curves #
            plot_metrics_jsonl(run_dir / "metrics.jsonl", run_dir / "curves.png")  # we plot #
        except Exception as e:  # we ignore plotting errors #
            print(f"plot_error={e}")  # we print #
        return run_dir  # we return #
    finally:  # we restore #
        import sys  # we import #
        sys.stdout = tee._stdout  # we restore #
        sys.stderr = tee._stderr  # we restore #
        tee.close()  # we close #


def main() -> None:  # we provide a default entrypoint #
    base = Path("experiments/helmholtz")  # we set base #
    run_dir = base / "runs" / f"run_{now_ts()}"  # we set run dir #
    dataset_cfg = HelmholtzDatasetConfig(n_grid=64, n_train=256, n_test=64)  # we set dataset #
    dataset_path = base / "data" / f"helmholtz_n{dataset_cfg.n_grid}_train{dataset_cfg.n_train}_test{dataset_cfg.n_test}.npz"  # we set path #
    if not dataset_path.exists():  # we generate if missing #
        generate_dataset_npz(dataset_path, dataset_cfg)  # we generate #

    io_cfg = HelmholtzIOConfig(use_coords=True, include_bmask=False)  # we set io #
    train_cfg = TrainConfig(seed=0, device="cuda", batch_size=8, num_epochs=20, lr=1e-3)  # we set train #

    model_cfg = ModelConfig(  # we set default model #
        model_type="deeponet",
        in_channels=4 + (2 if io_cfg.use_coords else 0) + (1 if io_cfg.include_bmask else 0),
        deeponet_latent_dim=128,
        trunk_fix_wb=True,
        branch_type="mlp",
        sensors_grid=16,
    )  # we set #

    dump_json(run_dir / "run_manifest.json", {"dataset_path": str(dataset_path)})  # we save manifest #
    run_one(run_dir, dataset_path, dataset_cfg, io_cfg, model_cfg, train_cfg)  # we run #


if __name__ == "__main__":  # we run main #
    main()  # we run #

