"""
Joint Mapping Utility
MediaPipe 33 landmarks <-> HumanSC3D 25 joints <-> Common 17 joints alignment.
"""

from typing import List, Tuple
import numpy as np

COMMON_JOINT_NAMES = [
    "Pelvis",         # 0: Root
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
    (0, 1), (1, 2), (2, 3),        # Right Leg
    (0, 4), (4, 5), (5, 6),        # Left Leg
    (0, 7), (7, 16), (16, 8),      # Spine / Torso
    (8, 9),                        # Head
    (16, 10), (10, 11), (11, 12),  # Left Arm
    (16, 13), (13, 14), (14, 15)   # Right Arm
]


def align_mediapipe_joints(mp_landmarks: np.ndarray) -> np.ndarray:
    """
    Convert MediaPipe (N, 33, 3) or (33, 3) landmarks into Common 17 joints.
    """
    is_batched = mp_landmarks.ndim == 3
    if not is_batched:
        mp_landmarks = mp_landmarks[np.newaxis, ...]

    B = mp_landmarks.shape[0]
    out = np.zeros((B, 17, 3), dtype=np.float32)

    l_hip = mp_landmarks[:, 23]
    r_hip = mp_landmarks[:, 24]
    pelvis = (l_hip + r_hip) * 0.5

    l_sh = mp_landmarks[:, 11]
    r_sh = mp_landmarks[:, 12]
    neck = (l_sh + r_sh) * 0.5
    thorax = neck * 0.7 + pelvis * 0.3
    spine = pelvis * 0.6 + thorax * 0.4

    out[:, 0] = pelvis
    out[:, 1] = r_hip
    out[:, 2] = mp_landmarks[:, 26] # Right Knee
    out[:, 3] = mp_landmarks[:, 28] # Right Ankle
    out[:, 4] = l_hip
    out[:, 5] = mp_landmarks[:, 25] # Left Knee
    out[:, 6] = mp_landmarks[:, 27] # Left Ankle
    out[:, 7] = spine
    out[:, 8] = neck
    out[:, 9] = mp_landmarks[:, 0]  # Nose / Head
    out[:, 10] = l_sh
    out[:, 11] = mp_landmarks[:, 13] # Left Elbow
    out[:, 12] = mp_landmarks[:, 15] # Left Wrist
    out[:, 13] = r_sh
    out[:, 14] = mp_landmarks[:, 14] # Right Elbow
    out[:, 15] = mp_landmarks[:, 16] # Right Wrist
    out[:, 16] = thorax

    return out if is_batched else out[0]


def align_humansc3d_joints(hsc_joints: np.ndarray) -> np.ndarray:
    """
    Convert HumanSC3D (N, 25, 3) or (25, 3) joints into Common 17 joints.
    HumanSC3D 25 joints topology (H36M standard):
      0: Pelvis
      1, 2, 3: Left Hip, Left Knee, Left Ankle
      4, 5, 6: Right Hip, Right Knee, Right Ankle
      7: Spine
      8: Thorax
      9: Neck
      10: Head
      11, 12, 13: Left Shoulder, Left Elbow, Left Wrist
      14, 15, 16: Right Shoulder, Right Elbow, Right Wrist
    """
    is_batched = hsc_joints.ndim == 3
    if not is_batched:
        hsc_joints = hsc_joints[np.newaxis, ...]

    B = hsc_joints.shape[0]
    out = np.zeros((B, 17, 3), dtype=np.float32)

    pelvis = hsc_joints[:, 0]
    spine = hsc_joints[:, 7]
    thorax = hsc_joints[:, 8]
    neck = hsc_joints[:, 9]
    head = hsc_joints[:, 10]

    out[:, 0] = pelvis
    out[:, 1] = hsc_joints[:, 4]   # Right Hip
    out[:, 2] = hsc_joints[:, 5]   # Right Knee
    out[:, 3] = hsc_joints[:, 6]   # Right Ankle
    out[:, 4] = hsc_joints[:, 1]   # Left Hip
    out[:, 5] = hsc_joints[:, 2]   # Left Knee
    out[:, 6] = hsc_joints[:, 3]   # Left Ankle
    out[:, 7] = spine
    out[:, 8] = neck
    out[:, 9] = head
    out[:, 10] = hsc_joints[:, 11] # Left Shoulder
    out[:, 11] = hsc_joints[:, 12] # Left Elbow
    out[:, 12] = hsc_joints[:, 13] # Left Wrist
    out[:, 13] = hsc_joints[:, 14] # Right Shoulder
    out[:, 14] = hsc_joints[:, 15] # Right Elbow
    out[:, 15] = hsc_joints[:, 16] # Right Wrist
    out[:, 16] = thorax

    return out if is_batched else out[0]


def get_skeleton_edges() -> List[Tuple[int, int]]:
    return SKELETON_EDGES
