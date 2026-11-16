# Team Workflow Guide: VS Code Copilot 협업 매뉴얼

본 문서는 **VS Code GitHub Copilot Chat** 환경에서 본 프로젝트의 에이전트 팀(PM, 3개 부서, 부장-부서원 계층 구조)을 효과적으로 가동하고, **토큰을 최대 80% 절약**하면서 **프로젝트가 산으로 가지 않도록 통제**하는 실전 가이드입니다.

---

## 1. 에이전트 조직도 및 역할 한눈에 보기

```
                             [사용자 (User)]
                                    │  (자연어 작업 지시)
                                    ▼
                        [Project Manager (PM)]
               (작업 분석, WBS 분할, 토큰 절약 프롬프트 정리)
             ┌──────────────────────┼──────────────────────┐
             ▼                      ▼                      ▼
    [Data Dept Manager]    [Model Dept Manager]     [QA Dept Manager]
      (데이터 부장)             (모델 부장)              (검증 부장)
             │                      │                      │
   ┌─────────┼─────────┐  ┌─────────┼─────────┐  ┌─────────┼─────────┐
   ▼         ▼         ▼  ▼         ▼         ▼  ▼         ▼         ▼
HumanSC3D MediaPipe Pipeline Arch    Loss/Opt Training Bench-  Visual-  Sanity
Specialist Specialist Engineer Spec    Spec   Engineer mark    ization  Auditor
```

---

## 2. VS Code Copilot Chat에서 에이전트 호출 방법

VS Code Copilot Chat 창(`Ctrl + Alt + I` 또는 사이드바)에서 다음 3가지 방법 중 하나로 대화를 시작할 수 있습니다:

### 방법 A: Prompt File 호출 (가장 추천)
VS Code의 프롬프트 파일 기능을 통해 사전 정의된 역할을 즉시 로드합니다:
- **PM에게 지시할 때**: `/pm <하고 싶은 일>` (또는 `@pm`)
- **데이터 작업만 직접 지시할 때**: `/dept-data <데이터 전처리 요청>`
- **모델 작업만 직접 지시할 때**: `/dept-model <모델 설계/학습 요청>`
- **평가/검증만 직접 지시할 때**: `/dept-qa <벤치마크 및 시각화 요청>`

### 방법 B: 파일 참조 멘션 (`#` 또는 `@`)
Copilot Chat 입력창에서 에이전트 문서를 직접 참조합니다:
- `#project_manager.md 이 지시를 바탕으로 각 부서에 작업 분할해줘: "s01 비디오에서 MediaPipe 랜드마크를 추출하고 캐싱해줘"`
- `#manager.md (Data Dept) 부서원들에게 파이프라인 제작 지시해줘`

---

## 3. 실전 대화 시나리오 예시

### 시나리오 1: 사용자가 PM에게 새로운 작업 요청
> **User**:  
> `train/s01` 데이터로 먼저 1-배치 오버피팅 테스트를 통과할 수 있는 미니 파이프라인을 만들어줘.

> **Project Manager (PM) 응답 예시**:
> ```markdown
> [PM Analysis] 사용자 요청: train/s01 기반 미니 파이프라인 및 1-배치 오버피팅 검증.
> WBS 1: Data Dept -> s01에서 비디오 1개, GT 1개 매칭하여 미니 DataLoader 생성
> WBS 2: Model Dept -> Residual Z-MLP 모델 설계 및 1-배치 오버피팅 루프 실행
> 
> ### 📋 [PM Directive] ➡️ Data Dept (수신: Data Dept Manager)
> - 목표: s01/001.mp4 및 GT joints3d_25/001.json 대상 미니 배치(16 frames) 추출
> - 계약: (B, 17, 3) 텐서, Pelvis root-relative 정규화
> 
> (Data 부장 검수 대기 중...)
> ```

---

### 시나리오 2: 부서 내 협업 및 부장의 게이트키핑 (데이터 부서)
> **Data Dept Manager**:  
> "MediaPipe Specialist는 `001.mp4`에서 랜드마크 추출하고, HumanSC3D Specialist는 `001.json`과 카메라 파라미터로 GT 카메라 좌표계 변환 수행하세요. 결과는 Pipeline Engineer가 취합합니다."

> **HumanSC3D Specialist**:
> ```markdown
> ### [TASK COMPLETE] HumanSC3D Specialist
> - 산출 파일: `src/data/hsc_parser.py`
> - 입출력 규격: In JSON -> Out (354, 17, 3) in camera frame
> - 자체 검증: s01/001.json 파싱 완료 (Pass)
> ```

> **Data Dept Manager (검수)**:
> ```markdown
> [Gatekeeper Check]
> [V] Pelvis index 0이 [0,0,0]인가? -> 확인 완료
> [V] Common 17 joints 매핑 사용했는가? -> 확인 완료
> 
> ### [DEPT SIGN-OFF] Data Engineering Dept
> - 핵심 산출물: `src/data/mini_dataloader.py`
> - 검수 결과: 5개 체크리스트 전원 통과. Model Dept로 인계 준비 완료.
> ```

---

### 시나리오 3: 산으로 갈 뻔한 상황을 부장이 반려하는 예시 (Anti-Drift)
만약 모델 부서원이 실수로 3D 전체 [X, Y, Z]를 다 예측하려고 시도한 경우:

> **Architecture Specialist**:  
> "더 높은 정확도를 위해 모델이 XY 좌표도 함께 미세조정하도록 `forward`를 작성했습니다."

> **Model Dept Manager (즉시 반려)**:
> ```markdown
> ### [REJECT & FIX] Model Dept Manager
> - 위반 항목: PROJECT_SPEC 제1조 'XY 좌표 불변의 원칙' 위반.
> - 수정 지시: 모델 출력은 오직 Z(또는 Delta Z) (B, 17, 1)로 한정하고, XY는 MediaPipe 출력을 그대로 통과(Identity)시키도록 즉시 수정할 것.
> ```

---

## 4. 토큰을 아끼는 3대 질문 팁 (Token-Saving Tips)

1. **"코드를 처음부터 다 보여줘" 대신 ➡️ "변경된 함수만 diff로 보여줘"**
   - 수백 줄의 파일 재출력을 방지하여 수천 개의 토큰을 절약합니다.
2. **"이거 어떻게 생각해?" 대신 ➡️ "PM 지침 양식으로 부서원 작업 지시서 작성해줘"**
   - 사설이나 잡담 없이 즉시 실행 가능한 마크다운 명세로 답변받을 수 있습니다.
3. **진행 상황 확인 시 ➡️ "각 부장 최종 검수 결과표(Template 3)만 보여줘"**
   - 8줄 이내의 압축 표로 전체 진행 상태를 파악할 수 있습니다.

---

## 5. 프로젝트가 산으로 가지 않게 하는 불변 수칙

1. **절대 규칙**: 모델은 **오직 Z축(또는 $\Delta Z$)만 예측**해야 합니다. (XY는 온전히 MediaPipe의 몫)
2. **좌표계 규칙**: 모든 관절의 중심은 골반(Pelvis)이며, 골반 좌표는 항상 `[0, 0, 0]`이어야 합니다.
3. **평가 규칙**: 모델의 성패는 **MediaPipe 순정 추론 대비 Z-MAE 개선율(%)**로만 평가합니다.
4. **학습 규칙**: 전체 데이터를 돌리기 전, 반드시 **1-배치 오버피팅 테스트(Sanity Overfitting)**를 통과해야 합니다.
