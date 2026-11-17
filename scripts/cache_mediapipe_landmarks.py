"""Extract and cache MediaPipe Pose world landmarks for every dataset video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skills.coordinate_transform.coord_utils import normalize_root_relative  # noqa: E402
from skills.joint_topology.joint_mapping import align_mediapipe_joints  # noqa: E402


def extract_video(video_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    landmarks: list[np.ndarray] = []
    visibility: list[np.ndarray] = []
    valid: list[bool] = []
    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        enable_segmentation=False,
        smooth_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.pose_world_landmarks is None:
                landmarks.append(np.full((33, 3), np.nan, dtype=np.float32))
                visibility.append(np.full(33, np.nan, dtype=np.float32))
                valid.append(False)
            else:
                points = result.pose_world_landmarks.landmark
                landmarks.append(
                    np.asarray(
                        [[point.x, point.y, point.z] for point in points],
                        dtype=np.float32,
                    )
                )
                visibility.append(
                    np.asarray([point.visibility for point in points], dtype=np.float32)
                )
                valid.append(True)
            frame_index += 1
    capture.release()

    if not landmarks:
        raise RuntimeError("Video contains no readable frames")
    return np.stack(landmarks), np.stack(visibility), np.asarray(valid, dtype=bool)


def cache_video(video_path: Path, cache_path: Path, gt_path: Path | None) -> None:
    world_landmarks, visibility, valid = extract_video(video_path)
    common17 = np.stack(
        [
            normalize_root_relative(align_mediapipe_joints(frame)).astype(np.float32)
            for frame in world_landmarks
        ]
    )
    gt_frames = -1
    if gt_path is not None and gt_path.exists():
        gt_frames = len(json.loads(gt_path.read_text())["joints3d_25"])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(cache_path.name + ".tmp")
    np.savez_compressed(
        temporary_path,
        world_landmarks_33=world_landmarks,
        visibility_33=visibility,
        common17_root_relative=common17,
        valid_mask=valid,
        missing_frame_indices=np.flatnonzero(~valid).astype(np.int32),
        video_frame_count=np.asarray(world_landmarks.shape[0], dtype=np.int32),
        gt_frame_count=np.asarray(gt_frames, dtype=np.int32),
    )
    generated_path = Path(str(temporary_path) + ".npz")
    generated_path.replace(cache_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["s01", "s02", "s03", "s06"],
        choices=["s01", "s02", "s03", "s06"],
    )
    parser.add_argument("--limit", type=int, default=0, help="Process at most N videos per subject")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    processed = 0
    skipped = 0
    failures: list[str] = []
    for subject in args.subjects:
        subject_root = ROOT / "train" / subject
        videos = sorted((subject_root / "videos").glob("*/*.mp4"))
        if args.limit:
            videos = videos[: args.limit]
        for video_path in videos:
            camera = video_path.parent.name
            action = video_path.stem
            cache_path = subject_root / "mediapipe_cache" / camera / f"{action}.npz"
            if cache_path.exists() and not args.overwrite:
                skipped += 1
                continue
            gt_path = subject_root / "joints3d_25" / f"{action}.json"
            try:
                cache_video(video_path, cache_path, gt_path)
                processed += 1
                print(f"[OK] {subject}/{camera}/{action}", flush=True)
            except Exception as error:
                failures.append(f"{subject}/{camera}/{action}: {error}")
                print(f"[FAIL] {failures[-1]}", flush=True)

    print(f"processed={processed} skipped={skipped} failures={len(failures)}")
    if failures:
        print("\n".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
