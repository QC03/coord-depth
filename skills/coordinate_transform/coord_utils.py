"""
Coordinate Transformation and Normalization Utilities.
Converts world coordinates to camera coordinates, applies root-relative normalization,
and aligns coordinate axes between MediaPipe and HumanSC3D.
"""

from typing import Union
import numpy as np
import torch


def world_to_camera_frame(
    joints_world: np.ndarray,
    R: np.ndarray,
    T: np.ndarray
) -> np.ndarray:
    """
    Transform 3D joints from World coordinate system to Camera coordinate system.
    Formula: X_cam = (R @ X_world.T).T + T

    Args:
        joints_world: (N, J, 3) or (J, 3) in meters
        R: (3, 3) rotation matrix
        T: (3,) or (1, 3) translation vector
    Returns:
        joints_cam: same shape as joints_world
    """
    is_batched = joints_world.ndim == 3
    if not is_batched:
        joints_world = joints_world[np.newaxis, ...]

    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    T = np.asarray(T, dtype=np.float32).reshape(1, 1, 3)

    # (N, J, 3) @ R.T + T
    joints_cam = np.matmul(joints_world, R.T) + T

    return joints_cam if is_batched else joints_cam[0]


def normalize_root_relative(
    joints: Union[np.ndarray, torch.Tensor],
    root_idx: int = 0
) -> Union[np.ndarray, torch.Tensor]:
    """
    Subtract root joint (Pelvis, index 0) position from all joints.
    After this, joints[..., root_idx, :] == [0, 0, 0].
    """
    if isinstance(joints, torch.Tensor):
        root = joints[..., root_idx:root_idx+1, :]
        return joints - root
    else:
        root = np.expand_dims(joints[..., root_idx, :], axis=-2)
        return joints - root


def compute_bone_lengths(
    joints: Union[np.ndarray, torch.Tensor],
    edges: list
) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute Euclidean lengths of bones specified by edges.
    Can be used for consistency loss or sanity checks.
    """
    if isinstance(joints, torch.Tensor):
        lengths = []
        for u, v in edges:
            diff = joints[..., u, :] - joints[..., v, :]
            dist = torch.norm(diff, dim=-1, keepdim=True)
            lengths.append(dist)
        return torch.cat(lengths, dim=-1)
    else:
        lengths = []
        for u, v in edges:
            diff = joints[..., u, :] - joints[..., v, :]
            dist = np.linalg.norm(diff, axis=-1, keepdims=True)
            lengths.append(dist)
        return np.concatenate(lengths, axis=-1)
