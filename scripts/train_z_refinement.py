"""Train the Z-only refinement model on cached HumanSC3D/MediaPipe pairs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skills.coordinate_transform.coord_utils import (  # noqa: E402
    compute_bone_lengths,
    normalize_root_relative,
    world_to_camera_frame,
)
from skills.evaluation_metrics.metrics import compute_z_mae  # noqa: E402
from skills.joint_topology.joint_mapping import (  # noqa: E402
    align_humansc3d_joints,
    get_skeleton_edges,
)


class ZResidualMLP(nn.Module):
    """Predict only a depth residual; MediaPipe XY is never predicted."""

    def __init__(self, joints: int = 17) -> None:
        super().__init__()
        self.joints = joints
        self.network = nn.Sequential(
            nn.Linear(joints * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, joints),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.network(inputs.reshape(inputs.shape[0], -1))
        return residual.reshape(inputs.shape[0], self.joints, 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_ground_truth(gt_path: Path, camera_path: Path) -> np.ndarray:
    raw = np.asarray(json.loads(gt_path.read_text())["joints3d_25"], dtype=np.float32)
    extrinsics = json.loads(camera_path.read_text())["extrinsics"]
    camera = world_to_camera_frame(
        raw,
        np.asarray(extrinsics["R"], dtype=np.float32),
        np.asarray(extrinsics["T"], dtype=np.float32),
    )
    return normalize_root_relative(align_humansc3d_joints(camera)).astype(np.float32)


def load_subject(subject: str) -> tuple[np.ndarray, np.ndarray]:
    subject_root = ROOT / "train" / subject
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    cache_paths = sorted((subject_root / "mediapipe_cache").glob("*/*.npz"))
    if not cache_paths:
        raise FileNotFoundError(f"No MediaPipe caches found for {subject}")

    gt_cache: dict[tuple[str, str], np.ndarray] = {}
    for cache_path in cache_paths:
        camera = cache_path.parent.name
        action = cache_path.stem
        gt_key = (camera, action)
        if gt_key not in gt_cache:
            gt_path = subject_root / "joints3d_25" / f"{action}.json"
            camera_path = subject_root / "camera_parameters" / camera / f"{action}.json"
            gt_cache[gt_key] = load_ground_truth(gt_path, camera_path)

        with np.load(cache_path) as cached:
            mp_common = np.asarray(cached["common17_root_relative"], dtype=np.float32)
            frame_count = min(mp_common.shape[0], gt_cache[gt_key].shape[0])
            valid = np.ones(frame_count, dtype=bool)
            if "valid_mask" in cached.files:
                valid &= np.asarray(cached["valid_mask"][:frame_count], dtype=bool)
            valid &= np.isfinite(mp_common[:frame_count]).all(axis=(1, 2))
            gt = gt_cache[gt_key][:frame_count]
            valid &= np.isfinite(gt).all(axis=(1, 2))
            if valid.any():
                inputs.append(mp_common[:frame_count][valid])
                targets.append(gt[valid])

    if not inputs:
        raise RuntimeError(f"No valid aligned frames found for {subject}")
    return np.concatenate(inputs), np.concatenate(targets)


def loss_components(
    pred_z: torch.Tensor,
    inputs: torch.Tensor,
    target_xyz: torch.Tensor,
    edges: list[tuple[int, int]],
    bone_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    refined_xyz = torch.cat((inputs[:, :, :2], pred_z), dim=-1)
    z_mae = torch.mean(torch.abs(pred_z - target_xyz[:, :, 2:3]))
    pred_bones = compute_bone_lengths(refined_xyz, edges)
    target_bones = compute_bone_lengths(target_xyz, edges)
    bone_loss = torch.mean(torch.abs(pred_bones - target_bones))
    return z_mae + bone_weight * bone_loss, z_mae, bone_loss


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    edges: list[tuple[int, int]],
    bone_weight: float,
    optimizer: torch.optim.Optimizer | None,
    clip_norm: float,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_z = total_bone = total_count = 0.0
    for inputs, target_xyz in loader:
        inputs = inputs.to(device)
        target_xyz = target_xyz.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            baseline_z = inputs[:, :, 2:3]
            pred_z = baseline_z + model(inputs)
            loss, z_mae, bone_loss = loss_components(
                pred_z, inputs, target_xyz, edges, bone_weight
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
        count = float(inputs.shape[0])
        total_loss += float(loss.item()) * count
        total_z += float(z_mae.item()) * count
        total_bone += float(bone_loss.item()) * count
        total_count += count
    return total_loss / total_count, total_z / total_count, total_bone / total_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-subjects", nargs="+", default=["s01", "s02"])
    parser.add_argument("--val-subjects", nargs="+", default=["s03"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--bone-weight", type=float, default=1.0)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, float | int | str]:
    set_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    device = torch.device("cpu")
    train_inputs, train_targets = zip(*(load_subject(s) for s in args.train_subjects))
    val_inputs, val_targets = zip(*(load_subject(s) for s in args.val_subjects))
    train_x = torch.from_numpy(np.concatenate(train_inputs))
    train_y = torch.from_numpy(np.concatenate(train_targets))
    val_x = torch.from_numpy(np.concatenate(val_inputs))
    val_y = torch.from_numpy(np.concatenate(val_targets))

    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y), batch_size=args.batch_size, shuffle=False
    )
    model = ZResidualMLP().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    edges = get_skeleton_edges()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_z_mae = float("inf")
    best_epoch = 0
    checkpoint_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

    with torch.no_grad():
        baseline_val_z_mae = compute_z_mae(val_x[:, :, 2:3], val_y[:, :, 2:3])
    print(
        f"train_frames={len(train_x)} val_frames={len(val_x)} "
        f"baseline_val_z_mae_mm={baseline_val_z_mae:.3f}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_z, train_bone = run_epoch(
            model, train_loader, device, edges, args.bone_weight, optimizer, args.clip_norm
        )
        with torch.no_grad():
            val_loss, val_z, val_bone = run_epoch(
                model, val_loader, device, edges, args.bone_weight, None, args.clip_norm
            )
        scheduler.step()
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"train_z_mae_mm={train_z * 1000:.3f} train_bone={train_bone:.6f} "
            f"val_loss={val_loss:.6f} val_z_mae_mm={val_z * 1000:.3f} "
            f"val_bone={val_bone:.6f}",
            flush=True,
        )
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_z_mae_mm": val_z * 1000.0,
            "config": checkpoint_config,
        }
        torch.save(state, args.checkpoint_dir / "latest.pth")
        if val_z < best_val_z_mae:
            best_val_z_mae = val_z
            best_epoch = epoch
            torch.save(state, args.checkpoint_dir / "best_model.pth")

    result = {
        "best_epoch": best_epoch,
        "best_val_z_mae_mm": best_val_z_mae * 1000.0,
        "baseline_val_z_mae_mm": baseline_val_z_mae,
        "train_frames": len(train_x),
        "val_frames": len(val_x),
        "best_checkpoint": str(args.checkpoint_dir / "best_model.pth"),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run(parse_args())
