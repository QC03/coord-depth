# MediaPipe XY + Neural Z-Depth Refinement

> **하이브리드 3D 인간 포즈 추정 파이프라인**: 2D(XY)는 MediaPipe가 온전히 추론하고, Z(Depth)는 MediaPipe의 거친 추론값에 전용 딥러닝 모델의 추론을 합하여 정밀한 3D 포즈를 완성합니다.

---

## 📁 디렉터리 구조

```
coord-depth/
├── .github/
│   ├── copilot-instructions.md      # VS Code Copilot 전역 행동 수칙 (SSOT & 프로토콜)
│   └── prompts/                     # VS Code Copilot 프롬프트 파일 (/pm, /dept-* 등)
│       ├── pm.prompt.md
│       ├── dept-data.prompt.md
│       ├── dept-model.prompt.md
│       └── dept-qa.prompt.md
├── agents/                          # 계층적 에이전트 역할 정의서
│   ├── pm/
│   │   └── project_manager.md       # 총괄 프로젝트 매니저 (PM)
│   ├── data_dept/                   # 데이터 엔지니어링 부서
│   │   ├── manager.md               # 데이터 부장 (게이트키퍼)
│   │   ├── humansc3d_specialist.md  # HumanSC3D 파싱 및 GT 변환
│   │   ├── mediapipe_specialist.md  # MediaPipe 랜드마크 추출
│   │   └── dataset_pipeline_engineer.md # Dataset 및 DataLoader 구축
│   ├── model_dept/                  # 모델 및 연구 부서
│   │   ├── manager.md               # 모델 부장 (게이트키퍼)
│   │   ├── architecture_specialist.md # Residual Z 신경망 설계
│   │   ├── loss_optimization_specialist.md # Z-MAE, MPJPE, Bone Loss
│   │   └── training_engineer.md     # 학습 루프 및 1-배치 오버피팅
│   └── qa_dept/                     # QA 및 검증 부서
│       ├── manager.md               # QA 부장 (게이트키퍼)
│       ├── benchmark_specialist.md  # Baseline vs Hybrid 정량 평가
│       ├── visualization_specialist.md # 3D 스켈레톤 비교 렌더링
│       └── sanity_auditor.md        # 산으로 가지 않게 하는 무결성 감사관
├── skills/                          # 재사용 가능한 도메인 스킬 및 코드
│   ├── joint_topology/              # MediaPipe 33 <-> HumanSC3D 25 매핑
│   │   ├── SKILL.md
│   │   └── joint_mapping.py
│   ├── coordinate_transform/        # 카메라 좌표계 변환 및 Root-relative 정규화
│   │   ├── SKILL.md
│   │   └── coord_utils.py
│   ├── token_saver/                 # 토큰 절약 보고 템플릿
│   │   ├── SKILL.md
│   │   └── report_templates.md
│   ├── anti_drift/                  # 이탈 방지 헌법 및 부장 체크리스트
│   │   ├── SKILL.md
│   │   └── gatekeeper_checklist.md
│   └── evaluation_metrics/          # Z-MAE, MPJPE, PA-MPJPE 계산
│       ├── SKILL.md
│       └── metrics.py
├── specs/
│   └── PROJECT_SPEC.md              # 단일 진실 공급원 (SSOT)
├── train/                           # HumanSC3D 데이터셋 (s01, s02, s03, s06)
├── TEAM_WORKFLOW_GUIDE.md           # Copilot 실전 사용 매뉴얼
└── README.md
```

---

## 👥 부서 및 에이전트 구성도

| 계층 | 부서 / 역할 | 담당 에이전트 | 주요 임무 |
|---|---|---|---|
| **Top** | 총괄 오케스트레이션 | **Project Manager (PM)** | 사용자 요구사항 분석, 프롬프트 정리, 작업 분할, 최종 검수 |
| **Dept 1** | 데이터 엔지니어링 | **Data Dept Manager (부장)** | 데이터 파이프라인 검수, 관절 정렬 및 좌표계 정합성 승인 |
| | 부서원 | HumanSC3D Specialist | 카메라 파라미터(R,T) 적용 및 3D GT 추출 |
| | 부서원 | MediaPipe Specialist | MediaPipe Pose 실행 및 랜드마크 추출 캐싱 |
| | 부서원 | Dataset Pipeline Engineer | PyTorch Dataset/DataLoader 제작, 피험자 분할 |
| **Dept 2** | 모델 및 연구 | **Model Dept Manager (부장)** | 모델 헌법(XY 고정, Z 단독 예측) 준수 검수, 학습 승인 |
| | 부서원 | Architecture Specialist | Residual Depth MLP, GCN 등 신경망 아키텍처 설계 |
| | 부서원 | Loss & Opt Specialist | Z-MAE, 3D MPJPE, Bone Length Loss 설계 |
| | 부서원 | Training Engineer | 1-배치 오버피팅 테스트 및 본 학습 루프 구축 |
| **Dept 3** | QA 및 품질 보증 | **QA Dept Manager (부장)** | 벤치마크 공정성 검수, 개선율 기준 미달 시 반려 |
| | 부서원 | Benchmark Specialist | MediaPipe 순정 vs 제안 하이브리드 정량 평가 |
| | 부서원 | Visualization Specialist | 3D 스켈레톤 비교 플롯 및 실패 사례(Worst-K) 분석 |
| | 부서원 | Sanity Auditor | 좌표계 뒤집힘, 관절 혼동, 데이터 누수 전수 감사 |

---

## ⚡ 빠른 시작 (Quick Start in VS Code Copilot)

1. **VS Code Copilot Chat** 열기 (`Ctrl + Alt + I`)
2. 슬래시 명령어로 PM 호출:
   ```
   /pm train/s01 데이터로 1-배치 오버피팅을 검증할 수 있는 미니 파이프라인과 모델을 구성해줘.
   ```
3. 자세한 대화 팁과 워크플로우는 [TEAM_WORKFLOW_GUIDE.md](file:///c:/Users/tjdwn/OneDrive/Desktop/Development/coord-depth/TEAM_WORKFLOW_GUIDE.md)를 참조하세요.

## 1-배치 오버피팅 검증

`train/s01` 영상에서 MediaPipe Pose를 추출하고, 카메라 변환된 HumanSC3D GT와 Common 17 관절로 정렬한 뒤 Z residual 모델만 1개 배치에 과적합합니다.

```powershell
python scripts\overfit_one_batch.py --subject s01 --camera 50591643 --action 001
```

성공 조건은 최종 MSE `<= 1e-5 m²`, 역전파 NaN/Inf 없음, 최종 출력의 XY 변화 `0`입니다. 전체 데이터 학습 전 sanity check 용도이며 모델은 XY를 예측하지 않습니다.
