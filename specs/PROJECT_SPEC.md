# PROJECT_SPEC: MediaPipe XY + Neural Z-Depth Refinement

> **문서 상태**: [ACTIVE] 단일 진실 공급원 (Single Source of Truth, SSOT)  
> **프로젝트 목표**: MediaPipe Pose의 정확한 2D(XY) 추론 좌표와 대략적인 Z 좌표를 입력받아, 정밀 딥러닝 모델이 Z 좌표를 보정하여 정확한 3D 포즈를 예측하는 하이브리드 파이프라인 구축.

---

## 1. 핵심 철학 및 불변 규칙 (Non-Negotiable Rules)

1. **파이썬 런타임 환경 (Python 3.10 Standard)**:
   - 본 프로젝트의 공식 파이썬 버전은 **Python 3.10.x**로 고정한다. (MediaPipe 및 PyTorch C++ 바이너리 호환성 보장)
   - 모든 스크립트 실행 및 패키지 관리는 프로젝트 루트의 `.venv` 가상환경을 기준으로 한다.
2. **XY 좌표 불변의 원칙**:
   - 2D 평면상의 XY 좌표는 **온전히 MediaPipe Pose의 추론 결과**를 신뢰하고 그대로 사용한다.
   - 신경망 모델은 XY를 변경하거나 다시 예측하지 않는다.
3. **Z축 예측 및 보정의 원칙**:
   - MediaPipe의 Z 좌표($Z_{mp}$)는 단안 영상 특성상 깊이 모호성과 스케일 왜곡을 지닌다.
   - 모델은 $XY_{mp}$와 $Z_{mp}$를 입력받아 **보정량 $\Delta Z$ (Residual)** 또는 **정밀한 $Z_{refined}$**를 출력한다.
   - 최종 3D 좌표: $P_{final} = [X_{mp}, Y_{mp}, Z_{mp} + \Delta Z]$ (또는 $[X_{mp}, Y_{mp}, Z_{pred}]$).
4. **토큰 절약 및 간결성 원칙**:
   - 모든 에이전트는 불필요한 대화형 장황함을 배제하고, 구조화된 명세(IPO: Input-Process-Output)와 diff 형식으로 소통한다.
5. **산으로 가지 않는 원칙 (Anti-Drift)**:
   - 데이터셋, 관절 정의, 좌표계 규격은 본 문서에 명시된 규칙을 절대 임의로 변경하지 않는다.
   - 변경이 필요할 경우 반드시 PM(Project Manager)의 승인 하에 본 문서를 먼저 갱신해야 한다.

---

## 2. 데이터셋 및 좌표계 규격

### 2.1 데이터셋: HumanSC3D
- **위치**: `train/` (s01, s02, s03, s06)
- **구성**:
  - `videos/<camera_id>/<action_id>.mp4`: RGB 동영상 (카메라별 멀티뷰)
  - `camera_parameters/<camera_id>/<action_id>.json`: Extrinsics (R, T) 및 Intrinsics (f, c, k, p)
  - `joints3d_25/<action_id>.json`: 25개 관절 3D Ground Truth 좌표 (프레임별 `(N_frames, 25, 3)`)
  - `smplx/<action_id>.json`: SMPL-X 파라미터 (transl, body_pose 등)

### 2.2 표준 좌표계 (Canonical Coordinate System)
- **기준 좌표계**: **Root-Relative Camera Coordinate System**
- **원점 (Root)**: Pelvis / Hip Center (Left Hip과 Right Hip의 중점, 좌표 `[0, 0, 0]`)
- **단위**: 미터 (Meter, m)
- **축 방향**:
  - $+X$: 관찰자 기준 오른쪽 (Subject의 왼쪽)
  - $+Y$: 관찰자 기준 아래쪽 (Downwards)
  - $+Z$: 카메라에서 피험자 쪽으로 들어가는 깊이 방향 (Forward / Depth)
- **GT 변환식**:
  $$\mathbf{X}_{cam} = \mathbf{R} \cdot \mathbf{X}_{world} + \mathbf{T}$$
  $$\mathbf{X}_{rel} = \mathbf{X}_{cam} - \mathbf{X}_{cam}^{root}$$

---

## 3. 관절 토폴로지 (Joint Topology)

MediaPipe(33개 관절)와 HumanSC3D(25개 관절) 간의 공통 유효 관절을 추출하여 **공통 17개 관절 (COCO/H36M 표준 호환)**으로 정렬하여 학습 및 평가를 진행한다.

| 공통 ID | 관절 명칭 (Joint Name) | MediaPipe Index | HumanSC3D Index (BODY_25) |
|---|---|---|---|
| 0 | Pelvis (Root) | Mid-Hip (calc: (23+24)/2) | 8 (MidHip) |
| 1 | Right Hip | 24 | 9 |
| 2 | Right Knee | 26 | 10 |
| 3 | Right Ankle | 28 | 11 |
| 4 | Left Hip | 23 | 12 |
| 5 | Left Knee | 25 | 13 |
| 6 | Left Ankle | 27 | 14 |
| 7 | Spine / Thorax | Mid-Shoulder/Hip calc | 1 (Neck) |
| 8 | Neck | Mid-Shoulder (calc: (11+12)/2) | 1 (Neck) |
| 9 | Head / Nose | 0 (Nose) | 0 (Nose) |
| 10 | Left Shoulder | 11 | 5 |
| 11 | Left Elbow | 13 | 6 |
| 12 | Left Wrist | 15 | 7 |
| 13 | Right Shoulder | 12 | 2 |
| 14 | Right Elbow | 14 | 3 |
| 15 | Right Wrist | 16 | 4 |
| 16 | Pelvis-Spine center | Midpoint (Pelvis-Neck) | - |

*(세부 인덱스 매핑 코드는 `skills/joint_topology/joint_mapping.py` 참조)*

---

## 4. 모델 텐서 입출력 규격 (Tensor Interface Contract)

### 4.1 Input Tensor: $\mathbf{X}_{in}$
- **Shape**: `(Batch_Size, Num_Joints, 3)` (기본 $J=17$)
- **채널 구성**:
  - `[:, :, 0]`: $X_{mp}$ (MediaPipe 정규화 또는 Root-relative X)
  - `[:, :, 1]`: $Y_{mp}$ (MediaPipe 정규화 또는 Root-relative Y)
  - `[:, :, 2]`: $Z_{mp}$ (MediaPipe Rough Z)
- **추가 옵션 Feature (선택적)**:
  - Bone Length Prior / Confidence Score: `(Batch_Size, Num_Joints, 1)`

### 4.2 Output Tensor: $\hat{\mathbf{Z}}$ 또는 $\Delta \mathbf{Z}$
- **Shape**: `(Batch_Size, Num_Joints, 1)`
- **해석**:
  - Residual Mode (권장): $\mathbf{Z}_{refined} = \mathbf{Z}_{mp} + \Delta \mathbf{Z}$
  - Direct Mode: $\mathbf{Z}_{refined} = \hat{\mathbf{Z}}$

### 4.3 Target Tensor: $\mathbf{Z}_{gt}$
- **Shape**: `(Batch_Size, Num_Joints, 1)` (Root-relative Ground Truth Z)

---

## 5. 핵심 평가 지표 (Key Performance Indicators)

1. **MPJPE (Mean Per Joint Position Error)**:
   - 밀리미터(mm) 단위 관절 위치 오차
   - 기준: $\frac{1}{J}\sum_{j=1}^{J} \|\hat{\mathbf{P}}_j - \mathbf{P}_j^{gt}\|_2$
2. **PA-MPJPE (Procrustes-Aligned MPJPE)**:
   - 강체 변환(Rigid Alignment) 후 오차 (mm)
3. **Z-MAE (Z-Axis Mean Absolute Error)**:
   - Z축 깊이 단독 오차: $\frac{1}{J}\sum_{j=1}^{J} |\hat{Z}_j - Z_j^{gt}|$
4. **Baseline Target**:
   - MediaPipe 순정 추론 대비 Z-MAE 30% 이상 개선 달성.

---

## 6. 팀 조직 및 파이프라인 요약

```
[User Request]
       │
       ▼
[Project Manager (PM)] ── (WBS 분할, 토큰 절약 프롬프트 전달)
       │
       ├──► [Data Dept (데이터 부서)]
       │        ├── 부장 (Manager) - 검수 및 승인
       │        └── 부서원: HumanSC3D, MediaPipe, Dataset Pipeline Specialist
       │
       ├──► [Model Dept (모델 부서)]
       │        ├── 부장 (Manager) - 검수 및 승인
       │        └── 부서원: Architecture, Loss & Optimization, Training Engineer
       │
       └──► [QA & Evaluation Dept (검증 부서)]
                ├── 부장 (Manager) - 검수 및 승인
                └── 부서원: Benchmark, Visualization, Sanity Auditor
```
