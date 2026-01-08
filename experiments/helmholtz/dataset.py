import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class HelmholtzIOConfig:  # we define which channels are used by models #
    use_coords: bool = True  # we include x,y channels #
    include_bmask: bool = False  # we optionally include boundary mask as channel #


class HelmholtzNPZDataset(Dataset):  # we provide a pytorch dataset wrapper #
    def __init__(self, npz_path: Path, split: str, io_cfg: HelmholtzIOConfig):  # we store arrays #
        self.npz_path = Path(npz_path)  # we store path #
        self.split = str(split)  # we store split #
        self.io_cfg = io_cfg  # we store io config #
        data = np.load(self.npz_path, allow_pickle=False)  # we load data #
        self.meta = json.loads(str(data["meta"]))  # we parse json #
        self.x = data["x"].astype(np.float32)  # we store x grid #
        self.y = data["y"].astype(np.float32)  # we store y grid #

        prefix = "train" if self.split == "train" else "test"  # we pick split #
        self.f = data[f"{prefix}_f"].astype(np.float32)  # we store forcing #
        self.k2 = data[f"{prefix}_k2"].astype(np.float32)  # we store k2 #
        self.g = data[f"{prefix}_g"].astype(np.float32)  # we store boundary forcing #
        self.u = data[f"{prefix}_u"].astype(np.float32)  # we store solution #
        self.mask = data[f"{prefix}_mask"].astype(np.float32)  # we store domain mask #
        self.bmask = data[f"{prefix}_bmask"].astype(np.float32)  # we store boundary mask #
        self.shape_id = data[f"{prefix}_shape_id"].astype(np.int64)  # we store shape id #

        self.n = int(self.f.shape[0])  # we store length #
        self.n_grid = int(self.f.shape[1])  # we store grid size #

    def __len__(self) -> int:  # we provide length #
        return self.n  # we return #

    def _stack_input(self, idx: int) -> torch.Tensor:  # we build input tensor (C,H,W) #
        chans = [  # we build standard channels #
            self.f[idx],
            self.k2[idx],
            self.g[idx],
            self.mask[idx],
        ]  # we set channels #
        if self.io_cfg.include_bmask:  # we optionally include boundary mask #
            chans.append(self.bmask[idx])  # we append #
        if self.io_cfg.use_coords:  # we include coordinates #
            chans.append(self.x)  # we append #
            chans.append(self.y)  # we append #
        inp = np.stack(chans, axis=0).astype(np.float32)  # we stack #
        return torch.from_numpy(inp)  # we return tensor #

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # we return one sample dict #
        x_inp = self._stack_input(int(idx))  # we build input #
        u = torch.from_numpy(self.u[int(idx)][None, :, :])  # we add channel dim #
        mask = torch.from_numpy(self.mask[int(idx)][None, :, :])  # we add channel dim #
        bmask = torch.from_numpy(self.bmask[int(idx)][None, :, :])  # we add channel dim #
        g = torch.from_numpy(self.g[int(idx)][None, :, :])  # we add channel dim #
        return {  # we return dict #
            "x": x_inp,
            "u": u,
            "mask": mask,
            "bmask": bmask,
            "g": g,
            "shape_id": torch.tensor(int(self.shape_id[int(idx)]), dtype=torch.long),
        }

