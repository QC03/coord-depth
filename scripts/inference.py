"""Run Z-only 3D pose inference on a webcam or an arbitrary MP4 file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_z_refinement import ZResidualMLP  # noqa: E402
from skills.joint_topology.joint_mapping import (  # noqa: E402
    align_mediapipe_joints,
    get_skeleton_edges,
)
from skills.coordinate_transform.coord_utils import normalize_root_relative  # noqa: E402


def load_model(checkpoint_path: Path) -> ZResidualMLP:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = ZResidualMLP()
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def detect_common17(
    pose: mp.solutions.pose.Pose, frame: np.ndarray
) -> np.ndarray | None:
    result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if result.pose_world_landmarks is None:
        return None
    world = np.asarray(
        [[point.x, point.y, point.z] for point in result.pose_world_landmarks.landmark],
        dtype=np.float32,
    )
    return normalize_root_relative(align_mediapipe_joints(world)).astype(np.float32)


def refine_pose(model: ZResidualMLP, common17: np.ndarray) -> np.ndarray:
    inputs = torch.from_numpy(common17[None])
    with torch.no_grad():
        refined_z = inputs[:, :, 2:3] + model(inputs)
    return np.concatenate((common17[:, :2], refined_z.numpy()[0]), axis=-1)


def draw_2d_overlay(frame: np.ndarray, common17: np.ndarray | None) -> np.ndarray:
    output = frame.copy()
    if common17 is None:
        cv2.putText(
            output,
            "Pose not detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return output
    height, width = output.shape[:2]
    points = {}
    for index, (x, y, _) in enumerate(common17):
        px = int(width * (0.5 + float(x) * 0.8))
        py = int(height * (0.5 + float(y) * 0.8))
        points[index] = (px, py)
        cv2.circle(output, (px, py), 4, (0, 200, 255), -1)
    for left, right in get_skeleton_edges():
        cv2.line(output, points[left], points[right], (0, 220, 0), 2)
    cv2.putText(
        output,
        "MediaPipe XY (fixed) + refined Z",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 220, 0),
        2,
        cv2.LINE_AA,
    )
    return output


def draw_3d_skeleton(
    canvas: np.ndarray,
    refined_xyz: np.ndarray | None,
    origin_x: int,
    origin_y: int,
    panel_width: int,
    panel_height: int,
) -> None:
    if refined_xyz is None:
        cv2.putText(
            canvas,
            "3D pose unavailable",
            (origin_x + 20, origin_y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return
    scale = min(panel_width, panel_height) * 0.32
    center_x = origin_x + panel_width // 2
    center_y = origin_y + panel_height // 2
    projected: dict[int, tuple[int, int]] = {}
    for index, (x, y, z) in enumerate(refined_xyz):
        perspective = 1.0 / max(0.35, 1.0 + float(z) * 0.8)
        px = int(center_x + float(x) * scale * perspective)
        py = int(center_y + float(y) * scale * perspective)
        projected[index] = (px, py)
    for left, right in get_skeleton_edges():
        cv2.line(canvas, projected[left], projected[right], (255, 160, 0), 2)
    for index, point in projected.items():
        cv2.circle(canvas, point, 4, (255, 220, 0), -1)
        if index == 0:
            cv2.circle(canvas, point, 7, (0, 0, 255), 2)
    cv2.putText(
        canvas,
        "Refined 3D (X,Y,Z)",
        (origin_x + 20, origin_y + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 220, 0),
        2,
        cv2.LINE_AA,
    )


def draw_axis_projection(
    canvas: np.ndarray,
    refined_xyz: np.ndarray | None,
    horizontal_index: int,
    vertical_index: int,
    origin_x: int,
    origin_y: int,
    panel_width: int,
    panel_height: int,
    title: str,
    horizontal_label: str,
    vertical_label: str,
) -> None:
    if refined_xyz is None:
        cv2.putText(
            canvas,
            f"{title} unavailable",
            (origin_x + 20, origin_y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return
    values = refined_xyz[:, [horizontal_index, vertical_index]]
    limit = max(float(np.max(np.abs(values))), 0.05)
    scale = min(panel_width, panel_height) * 0.38 / limit
    center_x = origin_x + panel_width // 2
    center_y = origin_y + panel_height // 2
    projected: dict[int, tuple[int, int]] = {}
    for index, (horizontal, vertical) in enumerate(values):
        projected[index] = (
            int(center_x + float(horizontal) * scale),
            int(center_y - float(vertical) * scale),
        )
    for left, right in get_skeleton_edges():
        cv2.line(canvas, projected[left], projected[right], (180, 180, 180), 2)
    for point in projected.values():
        cv2.circle(canvas, point, 4, (0, 220, 255), -1)
    cv2.putText(
        canvas,
        f"{title} ({horizontal_label},{vertical_label})",
        (origin_x + 20, origin_y + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"horizontal: {horizontal_label} (m)",
        (origin_x + 20, origin_y + panel_height - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"vertical: {vertical_label} (m)",
        (origin_x + 20, origin_y + panel_height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Input MP4 path")
    source.add_argument("--webcam", type=int, help="Webcam device index, usually 0")
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "checkpoints" / "best_model.pth"
    )
    parser.add_argument("--output-video", type=Path, help="Optional annotated MP4 path")
    parser.add_argument("--output-npz", type=Path, help="Optional 3D prediction NPZ path")
    parser.add_argument("--model-complexity", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames")
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if args.input is not None and not args.input.exists():
        raise FileNotFoundError(f"Input video not found: {args.input}")
    capture_source: str | int = str(args.input) if args.input else int(args.webcam)
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open input: {capture_source}")

    model = load_model(args.checkpoint)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    input_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    input_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    writer = None
    if args.output_video is not None:
        args.output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(args.output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (input_width * 2, input_height * 2),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to create output video: {args.output_video}")

    predictions: list[np.ndarray] = []
    valid: list[bool] = []
    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok or (args.max_frames and frame_index >= args.max_frames):
                break
            common17 = detect_common17(pose, frame)
            refined = refine_pose(model, common17) if common17 is not None else None
            if refined is None:
                predictions.append(np.full((17, 3), np.nan, dtype=np.float32))
                valid.append(False)
            else:
                predictions.append(refined)
                valid.append(True)
            if not args.no_display or writer is not None:
                display = draw_2d_overlay(frame, common17)
                canvas = np.zeros((input_height * 2, input_width * 2, 3), dtype=np.uint8)
                canvas[:input_height, :input_width] = display
                draw_3d_skeleton(
                    canvas,
                    refined,
                    input_width,
                    0,
                    input_width,
                    input_height,
                )
                draw_axis_projection(
                    canvas,
                    refined,
                    2,
                    1,
                    0,
                    input_height,
                    input_width,
                    input_height,
                    "Depth profile Z-Y",
                    "Z",
                    "Y",
                )
                draw_axis_projection(
                    canvas,
                    refined,
                    2,
                    0,
                    input_width,
                    input_height,
                    input_width,
                    input_height,
                    "Depth profile Z-X",
                    "Z",
                    "X",
                )
                if writer is not None:
                    writer.write(canvas)
                if not args.no_display:
                    cv2.imshow("MediaPipe XY + Refined Z 3D Pose", canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
            frame_index += 1

    capture.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    if args.output_npz is not None:
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output_npz,
            refined_xyz=np.stack(predictions),
            valid_mask=np.asarray(valid, dtype=bool),
            frame_count=np.asarray(len(predictions), dtype=np.int32),
        )
    print(
        f"processed_frames={len(predictions)} valid_frames={sum(valid)} "
        f"invalid_frames={len(valid) - sum(valid)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
