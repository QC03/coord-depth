# Skill: Evaluation Metrics (3D 포즈 및 깊이 평가 지표)

## 개요
본 프로젝트의 성공 여부를 정량적으로 평가하기 위한 표준 3D Pose 및 Z-Depth 전용 평가 지표 모음입니다.

## 핵심 평가 지표

### 1. Z-MAE (Z-Axis Mean Absolute Error, mm 단위)
- Z축 단독 평균 오차:
  $$\text{Z-MAE} = \frac{1000}{N \cdot J} \sum_{i=1}^N \sum_{j=1}^J |\hat{Z}_{i,j} - Z_{i,j}^{gt}|$$
- 본 프로젝트의 최우선 타겟 지표입니다.

### 2. MPJPE (Mean Per Joint Position Error, mm 단위)
- 3D 전체 관절 유클리드 거리 오차:
  $$\text{MPJPE} = \frac{1000}{N \cdot J} \sum_{i=1}^N \sum_{j=1}^J \|\hat{\mathbf{P}}_{i,j} - \mathbf{P}_{i,j}^{gt}\|_2$$
- 여기서 $\hat{\mathbf{P}} = [X_{mp}, Y_{mp}, Z_{refined}]$ 입니다.

### 3. PA-MPJPE (Procrustes-Aligned MPJPE, mm 단위)
- 회전, 평행이동, 스케일(Rigid Alignment)을 맞춘 후의 형상 왜곡 오차.

### 4. PCK (Percentage of Correct Keypoints @ threshold)
- 허용 오차 임계값(예: 50mm, 150mm) 내에 들어온 관절의 비율(%).
