# Token-Saving Report Templates (초압축 보고서 템플릿)

---

### [Template 1] 부서원 ➡️ 부장: 단위 작업 완료 보고 (최대 7줄)

```markdown
### [TASK COMPLETE] <부서원 명칭>
- **작업명**: <작업 한 줄 요약>
- **산출 파일**: `<path/to/file.py>` (함수/클래스: `ClassName`)
- **입출력 규격**: In `(B, 17, 3)` -> Out `(B, 17, 1)` (단위: meter)
- **자체 검증**: 더미 1-배치 테스트 완료 (Pass / Fail)
- **특이사항/블로커**: 없음 (또는 1줄 요약)
```

---

### [Template 2] 부장 ➡️ 부서원: 검수 반려 및 수정 지시 (최대 3줄)

```markdown
### [REJECT & FIX] <부장 명칭>
- **위반 항목**: <PROJECT_SPEC 또는 체크리스트 위반 사항 명시>
- **수정 지시**: <구체적으로 수정할 변수명, 축, 텐서 형상 지시>
```

---

### [Template 3] 부장 ➡️ PM: 부서 최종 검수 완료 보고 (최대 8줄)

```markdown
### [DEPT SIGN-OFF] <부서명>
- **담당 부장**: <부장 명칭>
- **완료된 마일스톤**: <마일스톤 명칭>
- **참여 부서원**: <부서원 목록>
- **Gatekeeper 체크리스트**: 5개 항목 전원 Pass [V]
- **핵심 산출물**: `<주요 파일 경로>`
- **핵심 지표/결과**: Baseline 대비 Z-MAE X.Xmm 개선
- **다음 부서 인계 상태**: 준비 완료 (Ready)
```

---

### [Template 4] PM ➡️ 사용자: 최종 마일스톤 보고 (최대 10줄)

```markdown
### 🎯 [PM Milestone Report] <작업명>
- **진행 상태**: 완료 (Approved by all Department Managers)
- **주요 변경 사항**:
  1. <핵심 변경점 1>
  2. <핵심 변경점 2>
- **핵심 지표 (Performance Metrics)**:
  | 항목 | MediaPipe Baseline | Hybrid Model (XY_mp + Z_pred) | 개선율 |
  |---|---|---|---|
  | MPJPE (mm) | -- | -- | -- |
  | Z-MAE (mm) | -- | -- | -- |
- **다음 권장 작업**: <1줄 제안>
```
