"""
PoseDepthDataset: PyTorch Dataset for MediaPipe XY + Neural Z-Depth Refinement.
Loads pre-cached MediaPipe features and pairs them with camera-frame, root-relative GT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from skills.coordinate_transform.coord_utils import (
    world_to_camera_frame,
    normalize_root_relative,
)
from skills.joint_topology.joint_mapping import align_humansc3d_joints


class PoseDepthDataset(Dataset):
    """
    Dataset that returns paired (input_xyz_mp, target_z_gt, meta).
    - input_xyz: (17, 3) in root-relative meters [X_mp, Y_mp, Z_mp]
    - target_z:  (17, 1) in root-relative meters [Z_gt]
    """

    def __init__(
        self,
        data_root: str | Path = "train",
        subjects: Optional[List[str]] = None,
        max_actions: Optional[int] = None,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.subjects = subjects or ["s01", "s02"]
        self.stride = stride

        self.samples: List[Tuple[Path, Path, Path, int]] = []
        self._gt_cache: Dict[str, np.ndarray] = {}
        self._cam_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        self._build_index(max_actions=max_actions)

    def _build_index(self, max_actions: Optional[int] = None) -> None:
        """Scan directories and build frame-level index."""
        for subj in self.subjects:
            subj_dir = self.data_root / subj
            cache_dir = subj_dir / "mediapipe_cache"
            gt_dir = subj_dir / "joints3d_25"
            cam_dir = subj_dir / "camera_parameters"

            if not cache_dir.exists():
                continue

            for cam_folder in sorted(cache_dir.iterdir()):
                if not cam_folder.is_dir():
                    continue
                camera_id = cam_folder.name

                npz_files = sorted(cam_folder.glob("*.npz"))
                if max_actions is not None:
                    npz_files = npz_files[:max_actions]

                for npz_path in npz_files:
                    action_id = npz_path.stem
                    gt_path = gt_dir / f"{action_id}.json"
                    cam_path = cam_dir / camera_id / f"{action_id}.json"

                    if not (gt_path.exists() and cam_path.exists()):
                        continue

                    # Read frame count from npz metadata
                    try:
                        with np.load(npz_path) as data:
                            num_frames = int(data["gt_frame_count"])
                            if num_frames <= 0:
                                num_frames = len(data["common17_root_relative"])
                    except Exception:
                        continue

                    for f_idx in range(0, num_frames, self.stride):
                        self.samples.append((npz_path, gt_path, cam_path, f_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_gt_frames(self, gt_path: Path, cam_path: Path) -> np.ndarray:
        """Cached loader for camera-frame root-relative GT."""
        cache_key = f"{gt_path}:{cam_path}"
        if cache_key in self._gt_cache:
            return self._gt_cache[cache_key]

        gt_raw = np.asarray(json.loads(gt_path.read_text())["joints3d_25"], dtype=np.float32)
        cam_info = json.loads(cam_path.read_text())["extrinsics"]
        R = np.asarray(cam_info["R"], dtype=np.float32)
        T = np.asarray(cam_info["T"], dtype=np.float32)

        gt_cam = world_to_camera_frame(gt_raw, R, T)
        gt_17 = align_humansc3d_joints(gt_cam)
        gt_rel = normalize_root_relative(gt_17).astype(np.float32)

        # Keep cache size bounded
        if len(self._gt_cache) > 200:
            self._gt_cache.clear()

        self._gt_cache[cache_key] = gt_rel
        return gt_rel

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        npz_path, gt_path, cam_path, frame_idx = self.samples[idx]

        # 1. Load MediaPipe feature
        with np.load(npz_path) as cache:
            mp_frames = cache["common17_root_relative"]
            actual_f = min(frame_idx, len(mp_frames) - 1)
            x_mp = mp_frames[actual_f].astype(np.float32) # (17, 3)

        # 2. Load GT frame
        gt_frames = self._load_gt_frames(gt_path, cam_path)
        actual_gt_f = min(frame_idx, len(gt_frames) - 1)
        gt_frame = gt_frames[actual_gt_f] # (17, 3)

        # Target Z is (17, 1)
        target_z = gt_frame[:, 2:3]

        return (
            torch.from_numpy(x_mp),       # (17, 3) [X_mp, Y_mp, Z_mp]
            torch.from_numpy(target_z),   # (17, 1) [Z_gt]
            torch.from_numpy(gt_frame),   # (17, 3) [X_gt, Y_gt, Z_gt] (for 3D loss & metrics)
        )


def create_dataloaders(
    data_root: str | Path = "train",
    batch_size: int = 64,
    num_workers: int = 0,
    stride: int = 2,
    max_actions_per_subj: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates standard (Train, Val, Test) DataLoaders according to subject split:
    - Train: s01, s02
    - Val:   s03
    - Test:  s06
    """
    train_ds = PoseDepthDataset(
        data_root=data_root,
        subjects=["s01", "s02"],
        stride=stride,
        max_actions=max_actions_per_subj,
    )
    val_ds = PoseDepthDataset(
        data_root=data_root,
        subjects=["s03"],
        stride=stride * 2,
        max_actions=max_actions_per_subj,
    )
    test_ds = PoseDepthDataset(
        data_root=data_root,
        subjects=["s06"],
        stride=stride * 2,
        max_actions=max_actions_per_subj,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
