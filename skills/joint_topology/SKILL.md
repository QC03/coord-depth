# Skill: Joint Topology & Mapping (관절 토폴로지 매핑)

## 개요
MediaPipe Pose(33개 랜드마크)와 HumanSC3D(25개 3D 관절)는 관절 수와 인덱스 체계가 서로 다릅니다.  
본 스킬은 두 데이터 소스 간의 공통 관절(Common 17 Joints, H36M/COCO 표준 호환)을 추출하고 정렬하는 역할을 담당합니다.

## 표준 공통 17개 관절 정의
```python
COMMON_17_JOINTS = [
    "Pelvis",        # 0: Root (Mid-Hip)
    "Right_Hip",     # 1
    "Right_Knee",    # 2
    "Right_Ankle",   # 3
    "Left_Hip",      # 4
    "Left_Knee",     # 5
    "Left_Ankle",    # 6
    "Spine",         # 7: Spine / Mid-Torso
    "Neck",          # 8: Neck / Mid-Shoulder
    "Head",          # 9: Nose / Head Top
    "Left_Shoulder", # 10
    "Left_Elbow",    # 11
    "Left_Wrist",    # 12
    "Right_Shoulder",# 13
    "Right_Elbow",   # 14
    "Right_Wrist",   # 15
    "Thorax"         # 16: Chest center
]
```

## 사용법
`skills/joint_topology/joint_mapping.py`의 함수를 사용합니다:
- `align_mediapipe_joints(mp_landmarks)`: MediaPipe 33개 관절 -> Common 17개 관절 `(17, 3)`
- `align_humansc3d_joints(hsc_joints25)`: HumanSC3D 25개 관절 -> Common 17개 관절 `(17, 3)`
- `get_common_skeleton_edges()`: 시각화 및 Bone Length 계산을 위한 뼈대 연결선(Edge) 리스트 반환
