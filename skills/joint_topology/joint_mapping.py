"""
Joint Mapping Utility
MediaPipe 33 landmarks <-> HumanSC3D 25 joints <-> Common 17 joints alignment.
"""

from typing import List, Tuple
import numpy as np
import torch

COMMON_JOINT_NAMES = [
    "Pelvis",         # 0
    "Right_Hip",      # 1
    "Right_Knee",     # 2
    "Right_Ankle",    # 3
    "Left_Hip",       # 4
    "Left_Knee",      # 5
    "Left_Ankle",     # 6
    "Spine",          # 7
    "Neck",           # 8
    "Head",           # 9
    "Left_Shoulder",  # 10
    "Left_Elbow",     # 11
    "Left_Wrist",     # 12
    "Right_Shoulder", # 13
    "Right_Elbow",    # 14
    "Right_Wrist",    # 15
    "Thorax"          # 16
]

SKELETON_EDGES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3),    # Right Leg
    (0, 4), (4, 5), (5, 6),    # Left Leg
    (0, 7), (7, 16), (16, 8),  # Spine / Torso
    (8, 9),                    # Head
    (16, 10), (10, 11), (11, 12), # Left Arm
    (16, 13), (13, 14), (14, 15)  # Right Arm
]


def align_mediapipe_joints(mp_landmarks: np.ndarray) -> np.ndarray:
    """
    Convert MediaPipe (N, 33, 3) or (33, 3) landmarks into Common 17 joints.
    Calculates virtual joints (Pelvis, Spine, Neck, Thorax) from existing landmarks.
    """
    is_batched = mp_landmarks.ndim == 3
    if not is_batched:
        mp_landmarks = mp_landmarks[np.newaxis, ...]

    B = mp_landmarks.shape[0]
    out = np.zeros((B, 17, 3), dtype=np.float32)

    # MediaPipe Indices:
    # 0: nose, 11: l_sh, 12: r_sh, 13: l_el, 14: r_el, 15: l_wr, 16: r_wr
    # 23: l_hip, 24: r_hip, 25: l_knee, 26: r_knee, 27: l_ank, 28: r_ank

    l_hip = mp_landmarks[:, 23]
    r_hip = mp_landmarks[:, 24]
    pelvis = (l_hip + r_hip) * 0.5

    l_sh = mp_landmarks[:, 11]
    r_sh = mp_landmarks[:, 12]
    neck = (l_sh + r_sh) * 0.5
    thorax = (neck * 0.7 + pelvis * 0.3)
    spine = (pelvis * 0.6 + thorax * 0.4)

    out[:, 0] = pelvis
    out[:, 1] = r_hip
    out[:, 2] = mp_landmarks[:, 26] # r_knee
    out[:, 3] = mp_landmarks[:, 28] # r_ank
    out[:, 4] = l_hip
    out[:, 5] = mp_landmarks[:, 25] # l_knee
    out[:, 6] = mp_landmarks[:, 27] # l_ank
    out[:, 7] = spine
    out[:, 8] = neck
    out[:, 9] = mp_landmarks[:, 0]  # nose / head
    out[:, 10] = l_sh
    out[:, 11] = mp_landmarks[:, 13] # l_el
    out[:, 12] = mp_landmarks[:, 15] # l_wr
    out[:, 13] = r_sh
    out[:, 14] = mp_landmarks[:, 14] # r_el
    out[:, 15] = mp_landmarks[:, 16] # r_wr
    out[:, 16] = thorax

    return out if is_batched else out[0]


def align_humansc3d_joints(hsc_joints: np.ndarray) -> np.ndarray:
    """
    Convert HumanSC3D (N, 25, 3) or (25, 3) joints into Common 17 joints.
    HumanSC3D's first 17 joints use its dataset topology:
    0: Neck, 1-3: right shoulder/elbow/wrist, 4-6: left shoulder/elbow/wrist,
    7: Spine, 8: MidHip, 9-11: right hip/knee/ankle,
    14-16: left hip/knee/ankle. The remaining joints are auxiliary markers.
    """
    is_batched = hsc_joints.ndim == 3
    if not is_batched:
        hsc_joints = hsc_joints[np.newaxis, ...]

    B = hsc_joints.shape[0]
    out = np.zeros((B, 17, 3), dtype=np.float32)

    pelvis = hsc_joints[:, 8]  # MidHip
    spine = hsc_joints[:, 7]  # Spine
    neck = hsc_joints[:, 0]  # Neck
    thorax = (neck * 0.7 + pelvis * 0.3)

    out[:, 0] = pelvis
    out[:, 1] = hsc_joints[:, 9]  # r_hip
    out[:, 2] = hsc_joints[:, 10] # r_knee
    out[:, 3] = hsc_joints[:, 11] # r_ank
    out[:, 4] = hsc_joints[:, 14] # l_hip
    out[:, 5] = hsc_joints[:, 15] # l_knee
    out[:, 6] = hsc_joints[:, 16] # l_ank
    out[:, 7] = spine
    out[:, 8] = neck
    out[:, 9] = neck  # HumanSC3D has no separate nose joint
    out[:, 10] = hsc_joints[:, 4] # l_sh
    out[:, 11] = hsc_joints[:, 5] # l_el
    out[:, 12] = hsc_joints[:, 6] # l_wr
    out[:, 13] = hsc_joints[:, 1] # r_sh
    out[:, 14] = hsc_joints[:, 2] # r_el
    out[:, 15] = hsc_joints[:, 3] # r_wr
    out[:, 16] = thorax

    return out if is_batched else out[0]


def get_skeleton_edges() -> List[Tuple[int, int]]:
    return SKELETON_EDGES
