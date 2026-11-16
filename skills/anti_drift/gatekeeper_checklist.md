# Gatekeeper Checklist (부장 승인 필수 체크리스트)

모든 부서의 부장(Department Manager)은 부서원의 산출물을 승인하고 PM에게 넘기기 전, 반드시 아래 5대 체크리스트를 통과해야 합니다.  
단 하나라도 `[ ]` 미달 시 즉시 부서원에게 반려([REJECT & FIX])해야 합니다.

---

### [Checklist 1] SSOT 계약 준수 여부
- [ ] 입력 텐서 형상이 `(B, 17, 3)` [X_mp, Y_mp, Z_mp] 규격을 엄격히 지키고 있는가?
- [ ] 출력 텐서가 오직 Z축 차원 `(B, 17, 1)` 또는 $\Delta Z$ 만을 반환하는가? (XY 변경 시 즉각 탈락)
- [ ] 단위가 일관되게 미터(m) 단위인가? (mm나 pixel이 섞이지 않았는가?)

### [Checklist 2] 관절 토폴로지 일관성
- [ ] MediaPipe와 HumanSC3D의 공통 17개 관절 매핑이 `skills/joint_topology/joint_mapping.py`를 거쳤는가?
- [ ] 골반(Pelvis)이 인덱스 0번이며, Root-relative 변환 후 좌표가 `[0, 0, 0]`인가?

### [Checklist 3] 데이터 격리 및 분할 원칙
- [ ] Train/Val/Test 분할이 비디오 프레임 단위가 아닌 **피험자(Subject: s01, s02 vs s03, s06)** 단위로 완전히 격리되었는가?
- [ ] 카메라 파라미터(Extrinsics R, T)가 올바른 순서로 적용되었는가?

### [Checklist 4] 1-배치 오버피팅 (Sanity Overfitting Test)
- [ ] 전체 데이터 학습 전, 1개 미니배치(예: 16프레임)에 대해 수렴(Loss -> 0)하는지 확인했는가?
- [ ] 역전파 시 NaN이나 Inf gradient가 발생하지 않는가?

### [Checklist 5] 토큰 최적화 및 간결성
- [ ] 보고서가 10줄 이내의 [IPO] 템플릿으로 압축 작성되었는가?
- [ ] 불필요한 전체 코드 재출력 없이 변경 사항(diff/시그니처)만 기술되었는가?
