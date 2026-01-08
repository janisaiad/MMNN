import json
import os
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class Tee:  # we implement a simple stdout/stderr tee #
    def __init__(self, file_path: Path):  # we open file #
        self.file_path = Path(file_path)  # we store path #
        self.file_path.parent.mkdir(parents=True, exist_ok=True)  # we ensure directory #
        self.file = open(self.file_path, "a", buffering=1)  # we open file #
        self._stdout = sys.stdout  # we store #
        self._stderr = sys.stderr  # we store #

    def write(self, data: str) -> None:  # we write to both #
        self._stdout.write(data)  # we write #
        self.file.write(data)  # we write #

    def flush(self) -> None:  # we flush #
        self._stdout.flush()  # we flush #
        self.file.flush()  # we flush #

    def close(self) -> None:  # we close #
        try:  # we try #
            self.file.close()  # we close #
        except Exception:  # we ignore #
            pass  # we pass #


def now_ts() -> str:  # we build a timestamp #
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())  # we return #


def dump_json(path: Path, obj: Any) -> None:  # we dump json safely #
    path = Path(path)  # we normalize #
    path.parent.mkdir(parents=True, exist_ok=True)  # we ensure #
    if is_dataclass(obj):  # we convert dataclass #
        payload = asdict(obj)  # we convert #
    else:  # we keep #
        payload = obj  # we keep #
    with open(path, "w") as f:  # we open #
        json.dump(payload, f, indent=2, sort_keys=True)  # we write #


def append_jsonl(path: Path, row: dict[str, Any]) -> None:  # we append a jsonl row #
    path = Path(path)  # we normalize #
    path.parent.mkdir(parents=True, exist_ok=True)  # we ensure #
    with open(path, "a") as f:  # we open #
        f.write(json.dumps(row) + "\n")  # we append #


def set_all_seeds(seed: int) -> None:  # we seed common rngs #
    import numpy as np  # we import local #
    import torch  # we import local #

    s = int(seed)  # we cast #
    os.environ["PYTHONHASHSEED"] = str(s)  # we set env #
    np.random.seed(s)  # we seed numpy #
    torch.manual_seed(s)  # we seed torch #
    if torch.cuda.is_available():  # we seed cuda #
        torch.cuda.manual_seed_all(s)  # we seed #

