"""Fine-tune Z refinement on the 9-joint virtual frontal-pose dataset."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_z_refinement import ZResidualMLP  # noqa: E402
from skills.coordinate_transform.coord_utils import normalize_root_relative  # noqa: E402
from skills.joint_topology.joint_mapping import align_mediapipe_joints  # noqa: E402


VIRTUAL_TO_COMMON = {0: 9, 1: 8, 2: 13, 3: 14, 4: 15, 5: 10, 6: 11, 7: 12, 8: 0}
SUPERVISED_JOINTS = tuple(VIRTUAL_TO_COMMON.values())
WRIST_JOINTS = (12, 15)


def load_clip(name: str, root: Path) -> tuple[np.ndarray, np.ndarray]:
    video_path = root / "videos" / f"{name}.mp4"
    coord_path = root / "jointCoord" / f"{name}.json"
    records = json.loads(coord_path.read_text(encoding="utf-8"))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open virtual video: {video_path}")

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        for record in records:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Video/JSON frame mismatch in {name}")
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.pose_world_landmarks is None:
                continue
            world = np.asarray(
                [[point.x, point.y, point.z] for point in result.pose_world_landmarks.landmark],
                dtype=np.float32,
            )
            mp_common = normalize_root_relative(align_mediapipe_joints(world)).astype(np.float32)

            virtual = np.asarray(record["keypoints_3d"], dtype=np.float32)
            virtual -= virtual[8]
            virtual[:, 1] *= -1.0
            target = np.full((17, 3), np.nan, dtype=np.float32)
            for virtual_index, common_index in VIRTUAL_TO_COMMON.items():
                target[common_index] = virtual[virtual_index]
            inputs.append(mp_common)
            targets.append(target)
    capture.release()
    if not inputs:
        raise RuntimeError(f"No valid MediaPipe frames in {name}")
    return np.stack(inputs), np.stack(targets)


def load_dataset(names: list[str], root: Path) -> tuple[np.ndarray, np.ndarray]:
    inputs, targets = zip(*(load_clip(name, root) for name in names))
    return np.concatenate(inputs), np.concatenate(targets)


def virtual_loss(
    pred_z: torch.Tensor,
    inputs: torch.Tensor,
    target_xyz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_z = target_xyz[:, :, 2]
    valid = torch.isfinite(target_z)
    errors = torch.abs(pred_z[:, :, 0] - torch.nan_to_num(target_z))
    joint_weights = torch.ones_like(errors)
    joint_weights[:, list(WRIST_JOINTS)] = 2.0
    z_loss = (errors * joint_weights * valid).sum() / (joint_weights * valid).sum().clamp_min(1.0)

    arm_pairs = ((10, 11), (11, 12), (13, 14), (14, 15))
    bone_losses = []
    for left, right in arm_pairs:
        pair_valid = valid[:, left] & valid[:, right]
        pred_length = torch.linalg.vector_norm(
            torch.cat((inputs[:, left, :2], pred_z[:, left]), dim=-1)
            - torch.cat((inputs[:, right, :2], pred_z[:, right]), dim=-1),
            dim=-1,
        )
        target_length = torch.linalg.vector_norm(
            target_xyz[:, left] - target_xyz[:, right], dim=-1
        )
        bone_losses.append(
            (torch.abs(pred_length - target_length) * pair_valid).sum()
            / pair_valid.sum().clamp_min(1.0)
        )
    return z_loss + 0.25 * torch.stack(bone_losses).mean(), z_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "best_model.pth")
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints" / "virtual_finetuned.pth")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, float | int | str]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    virtual_root = ROOT / "train" / "virtualData"
    names = sorted(path.stem for path in (virtual_root / "videos").glob("*.mp4"))
    if len(names) < 3:
        raise RuntimeError("Virtual dataset needs at least three clips")
    validation_names = [name for name in names if name in {"Jab15", "straight15"}]
    train_names = [name for name in names if name not in validation_names]
    train_x, train_y = load_dataset(train_names, virtual_root)
    val_x, val_y = load_dataset(validation_names, virtual_root)

    model = ZResidualMLP()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_inputs = torch.from_numpy(val_x)
    val_targets = torch.from_numpy(val_y)
    with torch.no_grad():
        baseline = val_inputs[:, :, 2:3]
        valid = torch.isfinite(val_targets[:, :, 2])
        baseline_mae = (
            torch.abs(baseline[:, :, 0] - torch.nan_to_num(val_targets[:, :, 2])) * valid
        ).sum() / valid.sum()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred_z = inputs[:, :, 2:3] + model(inputs)
            loss, _ = virtual_loss(pred_z, inputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred_z = val_inputs[:, :, 2:3] + model(val_inputs)
            val_loss, val_z_loss = virtual_loss(pred_z, val_inputs, val_targets)
        print(
            f"epoch={epoch:02d} val_loss={val_loss.item():.6f} "
            f"val_z_mae_mm={val_z_loss.item() * 1000:.3f}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "source_checkpoint": str(args.checkpoint),
            "train_clips": train_names,
            "validation_clips": validation_names,
            "baseline_virtual_z_mae_mm": baseline_mae.item() * 1000,
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        args.output,
    )
    return {
        "baseline_virtual_z_mae_mm": baseline_mae.item() * 1000,
        "output": str(args.output),
        "train_frames": len(train_x),
        "validation_frames": len(val_x),
    }


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
