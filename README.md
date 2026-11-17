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
├── src/                             # 메인 모듈형 소스 코드
│   ├── data/                        # Dataset, DataLoader, 캐시 파서
│   ├── models/                      # Z-Refinement 신경망 모델 및 복합 손실함수
│   ├── training/                    # Trainer 클래스 및 학습 파이프라인
│   └── evaluation/                  # 벤치마크 평가 및 시각화 모듈
├── scripts/                         # 독립 실행형 스크립트 및 1-배치 오버피팅
│   ├── overfit_one_batch.py
│   └── cache_mediapipe_landmarks.py
├── checkpoints/                     # 모델 가중치 저장소 (*.pth)
├── results/                         # 평가 보고서 및 3D 플롯 결과
│   ├── metrics/
│   └── visualizations/
├── agents/                          # 계층적 에이전트 역할 정의서
│   ├── pm/
│   ├── data_dept/
│   ├── model_dept/
│   └── qa_dept/
├── skills/                          # 재사용 가능한 도메인 스킬 및 코드
│   ├── joint_topology/
│   ├── coordinate_transform/
│   ├── token_saver/
│   ├── anti_drift/
│   └── evaluation_metrics/
├── specs/
│   └── PROJECT_SPEC.md              # 단일 진실 공급원 (SSOT)
├── docs/
│   └── TEAM_WORKFLOW_GUIDE.md       # 에이전트 협업 실전 매뉴얼
├── train/                           # HumanSC3D 데이터셋 및 mediapipe_cache
├── requirements.txt                 # 파이썬 의존성 패키지 목록
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
3. 자세한 대화 팁과 워크플로우는 [docs/TEAM_WORKFLOW_GUIDE.md](file:///c:/Users/tjdwn/OneDrive/Desktop/Development/coord-depth/docs/TEAM_WORKFLOW_GUIDE.md)를 참조하세요.

## 1-배치 오버피팅 검증

`train/s01` 영상에서 MediaPipe Pose를 추출하고, 카메라 변환된 HumanSC3D GT와 Common 17 관절로 정렬한 뒤 Z residual 모델만 1개 배치에 과적합합니다.

```powershell
python scripts\overfit_one_batch.py --subject s01 --camera 50591643 --action 001
```

성공 조건은 최종 MSE `<= 1e-5 m²`, 역전파 NaN/Inf 없음, 최종 출력의 XY 변화 `0`입니다. 전체 데이터 학습 전 sanity check 용도이며 모델은 XY를 예측하지 않습니다.

## 본 학습 (Z-MAE + Bone Length Loss)

캐시된 MediaPipe Common-17과 카메라 좌표계 HumanSC3D GT를 사용해 subject 단위로 학습합니다.

```powershell
python scripts\train_z_refinement.py --train-subjects s01 s02 --val-subjects s03 --checkpoint-dir checkpoints
```

Best 모델은 validation Z-MAE 기준으로 `checkpoints\best_model.pth`에 저장되고, 마지막 epoch는 `checkpoints\latest.pth`에 저장됩니다. 입력 `(B,17,3)`의 XY는 MediaPipe 값을 그대로 유지하며 모델 출력은 Z residual `(B,17,1)`뿐입니다.

## QA: s06 테스트 평가

```powershell
python scripts\evaluate_s06.py --checkpoint checkpoints\best_model.pth --results-dir results
```

전체 s06의 baseline/refined Z-MAE와 개선율은 `results\s06_metrics.json`, 대표 3D 비교 플롯은 `results\visualizations\s06_skeleton_comparison.png`에 생성됩니다.
