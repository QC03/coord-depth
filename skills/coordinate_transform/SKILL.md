# Skill: Coordinate Transformation & Normalization (좌표계 변환 및 정규화)

## 개요
3D 인간 포즈 추정에서 좌표계 불일치(Coordinate Mismatch)는 모델 학습 실패의 주원인입니다.  
본 스킬은 HumanSC3D의 월드 좌표계를 카메라 좌표계로 변환하고, 골반(Pelvis)을 원점(0,0,0)으로 하는 Root-Relative 좌표계로 통일하는 기능을 제공합니다.

## 핵심 수식 및 규칙

### 1. 월드 좌표계 -> 카메라 좌표계 (Extrinsics 변환)
카메라 파라미터 $R \in \mathbb{R}^{3 \times 3}$, $T \in \mathbb{R}^{3}$가 주어졌을 때:
$$X_{cam} = R \cdot X_{world} + T$$

### 2. Root-Relative 정규화 (골반 기준)
골반(Pelvis, index 0) 좌표를 $X_{root}$라 할 때:
$$X_{rel} = X - X_{root}$$
- 이렇게 변환하면 $X_{rel}^{root} = [0, 0, 0]$이 되며 카메라와의 절대 거리(전역 평행이동)와 무관하게 인체의 포즈 자체의 Z 깊이(상대적 깊이)에 집중할 수 있습니다.

### 3. MediaPipe 좌표 특성 및 정렬
- MediaPipe `pose_world_landmarks`는 미터 단위의 골반 중심 상대 좌표계입니다.
- 단, MediaPipe의 카메라 Z축 부호나 스케일이 데이터셋 카메라 기준과 다를 수 있으므로 정규화 및 축 일치가 필요합니다.
