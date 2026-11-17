"""
Visualize 3D skeletons from Front View (XY), Side View (ZY), and Top View (XZ).
Validates joint mapping between HumanSC3D GT and MediaPipe.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skills.coordinate_transform.coord_utils import world_to_camera_frame, normalize_root_relative
from skills.joint_topology.joint_mapping import (
    COMMON_JOINT_NAMES,
    SKELETON_EDGES,
    align_humansc3d_joints,
    align_mediapipe_joints,
)

ROOT = Path(__file__).resolve().parents[1]


def render_skeleton_views(
    gt_joints: np.ndarray,
    mp_joints: np.ndarray,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """
    Renders 3 orthographic views: Front View (X vs Y), Side View (Z vs Y), Top View (X vs Z).
    gt_joints: (17, 3) in root-relative meters
    mp_joints: (17, 3) in root-relative meters
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    # View specifications: (title, x_dim, y_dim, xlabel, ylabel, invert_y)
    # Camera coordinate system: +X right, +Y down, +Z forward (depth)
    views = [
        ("Front View (Coronal: XY)", 0, 1, "X (meters: Left <-> Right)", "Y (meters: Up <-> Down)", True),
        ("Side View (Sagittal: ZY)", 2, 1, "Z (meters: Forward/Depth)", "Y (meters: Up <-> Down)", True),
        ("Top View (Transverse: XZ)", 0, 2, "X (meters: Left <-> Right)", "Z (meters: Forward/Depth)", False),
    ]

    for ax, (view_title, dx, dy, xlabel, ylabel, invert_y) in zip(axes, views):
        # 1. Draw Ground Truth Skeleton (Green solid)
        for u, v in SKELETON_EDGES:
            ax.plot(
                [gt_joints[u, dx], gt_joints[v, dx]],
                [gt_joints[u, dy], gt_joints[v, dy]],
                color="#107C41",
                linewidth=2.5,
                alpha=0.85,
                label="Ground Truth (HumanSC3D)" if (u, v) == SKELETON_EDGES[0] else "",
            )

        # 2. Draw MediaPipe Skeleton (Red dashed)
        for u, v in SKELETON_EDGES:
            ax.plot(
                [mp_joints[u, dx], mp_joints[v, dx]],
                [mp_joints[u, dy], mp_joints[v, dy]],
                color="#D83B01",
                linewidth=1.8,
                linestyle="--",
                alpha=0.75,
                label="MediaPipe Baseline" if (u, v) == SKELETON_EDGES[0] else "",
            )

        # 3. Draw Joint Nodes
        ax.scatter(gt_joints[:, dx], gt_joints[:, dy], color="#107C41", s=45, zorder=5)
        ax.scatter(mp_joints[:, dx], mp_joints[:, dy], color="#D83B01", s=30, zorder=5)

        # 4. Joint Labels (Annotations) for GT
        for j_idx, (name, pt) in enumerate(zip(COMMON_JOINT_NAMES, gt_joints)):
            # Short label: e.g. "1:R_Hip", "10:L_Sh"
            short_name = (
                name.replace("Right_", "R_")
                .replace("Left_", "L_")
                .replace("Shoulder", "Sh")
                .replace("Elbow", "El")
                .replace("Wrist", "Wr")
                .replace("Ankle", "Ank")
            )
            ax.annotate(
                f"{j_idx}:{short_name}",
                (pt[dx], pt[dy]),
                fontsize=7.5,
                color="#0E4A24",
                fontweight="semibold",
                xytext=(3, 3),
                textcoords="offset points",
            )

        ax.set_title(view_title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        if invert_y:
            ax.invert_yaxis()  # Invert Y so up is up on screen
        ax.axis("equal")
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        f"Skeleton Multi-View Verification (GT vs MediaPipe) {title_suffix}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Multi-view plot saved to: {output_path}")


def main() -> None:
    subject = "s01"
    camera = "50591643"
    action = "001"
    frame_idx = 0

    # 1. Load GT and convert to Camera frame root-relative
    gt_json = json.load(open(ROOT / f"train/{subject}/joints3d_25/{action}.json"))["joints3d_25"][frame_idx]
    cam_json = json.load(open(ROOT / f"train/{subject}/camera_parameters/{camera}/{action}.json"))["extrinsics"]

    gt_world = np.array(gt_json, dtype=np.float32)
    R = np.array(cam_json["R"], dtype=np.float32)
    T = np.array(cam_json["T"], dtype=np.float32)

    gt_cam = world_to_camera_frame(gt_world, R, T)
    gt_common17 = align_humansc3d_joints(gt_cam)
    gt_rel = normalize_root_relative(gt_common17)

    # 2. Load MediaPipe raw world landmarks and align
    cache = np.load(ROOT / f"train/{subject}/mediapipe_cache/{camera}/{action}.npz")
    mp_raw_33 = cache["world_landmarks_33"][frame_idx]
    mp_common17 = align_mediapipe_joints(mp_raw_33)
    mp_rel = normalize_root_relative(mp_common17)

    # 3. Render and save
    out_img = ROOT / "results/visualizations/skeleton_multi_view_verification.png"
    render_skeleton_views(gt_rel, mp_rel, out_img, title_suffix=f"[{subject} | Cam {camera} | Frame {frame_idx}]")


if __name__ == "__main__":
    main()
