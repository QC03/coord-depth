"""Run the mandatory one-batch Z-depth overfitting sanity check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skills.coordinate_transform.coord_utils import (  # noqa: E402
    normalize_root_relative,
    world_to_camera_frame,
)
from skills.evaluation_metrics.metrics import compute_z_mae  # noqa: E402
from skills.joint_topology.joint_mapping import (  # noqa: E402
    align_humansc3d_joints,
    align_mediapipe_joints,
)


class ZResidualMLP(nn.Module):
    """Predict only a depth residual; XY never enters the output path."""

    def __init__(self, joints: int = 17) -> None:
        super().__init__()
        self.joints = joints
        self.network = nn.Sequential(
            nn.Linear(joints * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, joints),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.network(inputs.reshape(inputs.shape[0], -1))
        return residual.reshape(inputs.shape[0], self.joints, 1)


def load_media_pipe_features(video_path: Path, expected_frames: int) -> np.ndarray:
    """Extract root-relative MediaPipe world landmarks for every GT frame."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_frames < expected_frames:
        capture.release()
        raise RuntimeError(
            f"Video has fewer frames than GT for {video_path}: "
            f"video={video_frames}, GT={expected_frames}"
        )

    features: list[np.ndarray] = []
    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_index = 0
        while frame_index < expected_frames:
            ok, frame = capture.read()
            if not ok:
                break
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.pose_world_landmarks is None:
                raise RuntimeError(
                    f"MediaPipe found no pose in frame {frame_index} of {video_path}"
                )
            landmarks = np.array(
                [[point.x, point.y, point.z] for point in result.pose_world_landmarks.landmark],
                dtype=np.float32,
            )
            common = align_mediapipe_joints(landmarks)
            features.append(normalize_root_relative(common).astype(np.float32))
            frame_index += 1
    capture.release()

    if frame_index != expected_frames:
        raise RuntimeError(
            f"Unable to extract all GT-aligned frames from {video_path}: "
            f"extracted={frame_index}, GT={expected_frames}, video={video_frames}"
        )
    return np.stack(features)


def load_ground_truth(gt_path: Path, camera_path: Path) -> np.ndarray:
    """Load camera-frame, root-relative Common-17 GT in meters."""
    gt = np.asarray(json.loads(gt_path.read_text())["joints3d_25"], dtype=np.float32)
    camera = json.loads(camera_path.read_text())["extrinsics"]
    camera_joints = world_to_camera_frame(
        gt,
        np.asarray(camera["R"], dtype=np.float32),
        np.asarray(camera["T"], dtype=np.float32),
    )
    common = align_humansc3d_joints(camera_joints)
    return normalize_root_relative(common).astype(np.float32)


def run(args: argparse.Namespace) -> dict[str, float | int | bool]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)

    subject_root = ROOT / "train" / args.subject
    video_path = subject_root / "videos" / args.camera / f"{args.action}.mp4"
    gt_path = subject_root / "joints3d_25" / f"{args.action}.json"
    camera_path = subject_root / "camera_parameters" / args.camera / f"{args.action}.json"
    for path in (video_path, gt_path, camera_path):
        if not path.exists():
            raise FileNotFoundError(path)

    gt = load_ground_truth(gt_path, camera_path)
    mp_features = load_media_pipe_features(video_path, len(gt))
    inputs = torch.from_numpy(mp_features[: args.batch_size])
    target_z = torch.from_numpy(gt[: args.batch_size, :, 2:3])
    model = ZResidualMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        baseline_z = inputs[:, :, 2:3]
        baseline_mae_mm = compute_z_mae(baseline_z, target_z)

    initial_loss = float("nan")
    final_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        delta_z = model(inputs)
        refined_z = baseline_z + delta_z
        loss = loss_fn(refined_z, target_z)
        if epoch == 1:
            initial_loss = float(loss.item())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {loss.item()}")
        loss.backward()
        gradients_finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        if not gradients_finite:
            raise FloatingPointError(f"Non-finite gradient at epoch {epoch}")
        optimizer.step()
        final_loss = float(loss.item())

    with torch.no_grad():
        delta_z = model(inputs)
        refined_z = baseline_z + delta_z
        final_xyz = torch.cat((inputs[:, :, :2], refined_z), dim=-1)
        xy_error = float(torch.max(torch.abs(final_xyz[:, :, :2] - inputs[:, :, :2])).item())
        final_mae_mm = compute_z_mae(refined_z, target_z)

    result: dict[str, float | int | bool] = {
        "subject": args.subject,
        "camera": args.camera,
        "action": args.action,
        "frames_in_video": len(gt),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "initial_mse_m2": initial_loss,
        "final_mse_m2": final_loss,
        "baseline_z_mae_mm": baseline_mae_mm,
        "refined_z_mae_mm": final_mae_mm,
        "max_xy_change": xy_error,
        "xy_invariant": xy_error == 0.0,
        "overfit_pass": final_loss <= args.loss_threshold,
    }
    if not result["overfit_pass"]:
        raise RuntimeError(
            f"One-batch overfitting failed: final MSE {final_loss:.8g} "
            f"> threshold {args.loss_threshold:.8g}"
        )
    if not result["xy_invariant"]:
        raise RuntimeError(f"XY invariant failed: max change {xy_error}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="s01")
    parser.add_argument("--camera", default="50591643")
    parser.add_argument("--action", default="001")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--loss-threshold", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
