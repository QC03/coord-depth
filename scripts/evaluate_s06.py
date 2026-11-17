"""Evaluate the trained Z-refinement model on the held-out s06 subject."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_z_refinement import (  # noqa: E402
    ZResidualMLP,
    load_ground_truth,
    load_subject,
)
from skills.evaluation_metrics.metrics import compute_z_mae  # noqa: E402
from skills.joint_topology.joint_mapping import get_skeleton_edges  # noqa: E402


def predict(model: ZResidualMLP, inputs: np.ndarray, batch_size: int) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(inputs[start : start + batch_size])
            refined_z = batch[:, :, 2:3] + model(batch)
            outputs.append(refined_z.numpy())
    return np.concatenate(outputs)


def first_valid_example() -> tuple[str, str, int, np.ndarray, np.ndarray]:
    subject_root = ROOT / "train" / "s06"
    for cache_path in sorted((subject_root / "mediapipe_cache").glob("*/*.npz")):
        camera = cache_path.parent.name
        action = cache_path.stem
        gt_path = subject_root / "joints3d_25" / f"{action}.json"
        camera_path = subject_root / "camera_parameters" / camera / f"{action}.json"
        gt = load_ground_truth(gt_path, camera_path)
        with np.load(cache_path) as cached:
            mp = np.asarray(cached["common17_root_relative"], dtype=np.float32)
            count = min(len(mp), len(gt))
            valid = np.ones(count, dtype=bool)
            if "valid_mask" in cached.files:
                valid &= np.asarray(cached["valid_mask"][:count], dtype=bool)
            valid &= np.isfinite(mp[:count]).all(axis=(1, 2))
            valid &= np.isfinite(gt[:count]).all(axis=(1, 2))
            indices = np.flatnonzero(valid)
            if len(indices):
                index = int(indices[0])
                return camera, action, index, mp[index], gt[index]
    raise RuntimeError("No valid s06 frame available for visualization")


def plot_skeletons(
    mp_xyz: np.ndarray,
    refined_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    figure = plt.figure(figsize=(15, 5))
    edges = get_skeleton_edges()
    panels = [
        ("MediaPipe baseline", mp_xyz, "tab:orange"),
        ("Refined model", refined_xyz, "tab:blue"),
        ("Ground truth", gt_xyz, "tab:green"),
    ]
    for panel_index, (name, points, color) in enumerate(panels, start=1):
        axis = figure.add_subplot(1, 3, panel_index, projection="3d")
        for left, right in edges:
            axis.plot(
                points[[left, right], 0],
                points[[left, right], 1],
                points[[left, right], 2],
                color=color,
                linewidth=2,
            )
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, s=22)
        axis.set_title(name)
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.set_zlabel("Z (m)")
        axis.view_init(elev=15, azim=-70)
    figure.suptitle(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "best_model.pth")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, float | int | str]:
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = ZResidualMLP()
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    inputs, target = load_subject("s06")
    refined_z = predict(model, inputs, args.batch_size)
    baseline_z = inputs[:, :, 2:3]
    baseline_mae = compute_z_mae(baseline_z, target[:, :, 2:3])
    refined_mae = compute_z_mae(refined_z, target[:, :, 2:3])
    improvement = (baseline_mae - refined_mae) / baseline_mae * 100.0

    camera, action, frame, mp_frame, gt_frame = first_valid_example()
    refined_frame = np.concatenate(
        (mp_frame[:, :2], predict(model, mp_frame[None], 1)[0]), axis=-1
    )
    plot_path = args.results_dir / "visualizations" / "s06_skeleton_comparison.png"
    plot_skeletons(
        mp_frame,
        refined_frame,
        gt_frame,
        plot_path,
        f"s06/{camera}/{action}, frame {frame}",
    )
    metrics = {
        "subject": "s06",
        "frames": len(inputs),
        "checkpoint": str(args.checkpoint),
        "baseline_z_mae_mm": baseline_mae,
        "refined_z_mae_mm": refined_mae,
        "improvement_percent": improvement,
        "visualization": str(plot_path),
        "visualization_camera": camera,
        "visualization_action": action,
        "visualization_frame": frame,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "s06_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    run(parse_args())
