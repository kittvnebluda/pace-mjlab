# Author: Grigorii Sizikov
# Licensed under the Apache License 2.0

"""Print the latest CMA-ES mean of a fit run, stage 1 (PACE parameters) or stage 2 (motor constants)."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from pace_mjlab import PROJECT_ROOT
from pace_mjlab.config import aliengo_pace_cfg


@dataclass
class Args:
    run: Path | None = None
    """Run directory; defaults to the newest run of --stage."""
    stage: Literal[1, 2] = 1
    """Stage whose newest run is shown when --run is not given."""


def _latest_mean(run: Path) -> Path:
    means = sorted(run.glob("mean_*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
    if not means:
        raise FileNotFoundError(f"no mean_*.pt in {run}")
    return means[-1]


def main() -> None:
    args = tyro.cli(Args)
    cfg = aliengo_pace_cfg()
    run = args.run
    if run is None:
        log_dir = PROJECT_ROOT / "logs" / "pace" / cfg.robot_name
        if args.stage == 2:
            log_dir = log_dir / "stage2"
        runs = sorted(p for p in log_dir.iterdir() if p.is_dir() and any(p.glob("mean_*.pt")))
        if not runs:
            raise FileNotFoundError(f"no runs in {log_dir}")
        run = runs[-1]

    mean_file = _latest_mean(run)
    params = torch.load(mean_file, map_location="cpu").float()
    print(f"[INFO]: {mean_file}")

    joint_order = cfg.joint_order
    n = len(joint_order)
    groups = list(cfg.envelope.groups)
    if params.numel() == 4 * n + 1:
        print(f"{'joint':16s} {'armature':>10s} {'viscous':>10s} {'coulomb':>10s} {'bias':>10s}")
        for j, name in enumerate(joint_order):
            print(
                f"{name:16s} {params[j]:10.4f} {params[n + j]:10.4f} {params[2 * n + j]:10.4f} "
                f"{params[3 * n + j]:+10.4f}"
            )
        delay = params[4 * n].item()
        print(f"delay {delay:.3f} ({int(delay)} steps applied)")
    elif params.numel() == 2 * len(groups):
        for g, name in enumerate(groups):
            print(f"{name:6s} c_omega {params[g]:.4f} V s/rad, c_tau {params[len(groups) + g]:.4f} V/(N m)")
    else:
        raise ValueError(f"{mean_file} has {params.numel()} values, expected {4 * n + 1} or {2 * len(groups)}")
