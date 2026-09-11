# ADR-0001. 포즈 백엔드 선택

| 항목 | 값 |
| --- | --- |
| 상태 | **제안 (Proposed)** — 확정은 M2 벤치마크 이후 |
| 작성일 | 2026-09-11 |
| 결정자 | 개발자(본인) |
| 관련 | [PROJECT.md](../PROJECT.md) Q1·A1·R2·R3·R7·R8, [ARCHITECTURE.md](../ARCHITECTURE.md) §5.1·§5.2·§13 |
| 관련 문서 | [licenses.md](../licenses.md) |

---

## 1. 맥락

ARCHITECTURE §5.1은 포즈 추정을 3개 티어(A 경량 2D / B 2D+리프팅 / C world-grounded)로 두고 어댑터 뒤에 숨기기로 했으며,
"어느 것도 지금 확정하지 않는다"(PROJECT.md Q1)가 현재 상태다. 이 ADR은 **후보를 실제 배포처·라이선스 단위로 특정**하고,
M2에서 수행할 선정 절차를 고정한다.

조사 시점(2026-09-11) 기준으로, 후보를 Hugging Face Hub에서 받을 수 있는지 여부까지 확인했다.
재현성(NFR-03)과 `scripts/download_models.py`의 단순함이 이 결정의 숨은 평가 축이다.

---

## 2. 결정을 실질적으로 좌우하는 제약 3개

조사 결과 드러난, 티어 비교보다 먼저 걸리는 제약들이다.

### C1. 키포인트 포맷에 발 관절이 있어야 한다

DVI-Canonical(§4.3)은 `heel_l/r`, `foot_index_l/r`를 요구한다. 그런데 **COCO-17에는 발 관절이 없다**
(발목까지만 있음). 이를 파생 규칙으로 채울 수 없다 — 발끝과 뒤꿈치는 발목에서 유도되지 않는다.

영향받는 지표:

| 지표 | 필요 관절 | COCO-17로 가능한가 |
| --- | --- | --- |
| AXI-05 접지 품질 (발끝→뒤꿈치 순서) | `heel`, `foot_index` | **불가** |
| AXI-01 무게중심 오프셋 (지지발 접지점) | `heel`, `foot_index` | 발목 근사만 (정확도 하락) |
| AXI-06 축 수직성 | `ankle` | 가능 |

→ **결론: 2D 포맷은 Halpe-26 또는 COCO-WholeBody-133로 시작한다.**
PROJECT.md A1("DVI-Canonical을 H36M-17 기반 vs SMPL 기반")에 대한 제안 답은 **둘 다 아니고 Halpe-26 기반**이다.
H36M-17도 발 관절이 없어 같은 문제를 갖는다.

### C2. Tier B의 3D 리프터는 배포 상태가 가장 나쁘다

| 구성 요소 | 배포처 | 재현성 |
| --- | --- | --- |
| 고정확도 2D (ViTPose, RTMPose) | HF 공식/준공식 | 양호 |
| **시간축 3D 리프터 (MotionBERT 계열)** | **GitHub + 외부 스토리지, HF 공식 배포 없음** | **나쁨** |

Tier B가 "오프라인 기본"이라는 §5.1의 초기 가정과 달리, **가중치 조달 관점에서는 Tier C보다 불편하다.**
이것만으로 Q1이 Tier C로 기울 근거가 된다.

### C3. Top-down 추정기는 클로즈 엠브레이스에서 구조적으로 취약하다

두 사람의 바운딩박스가 거의 완전히 겹치면, 박스 안에 인물이 둘이므로 top-down 모델은
"어느 쪽 관절을 낼지" 결정할 수 없다. PROJECT.md R1의 완화책(신뢰도 게이팅 + 시간축 보간)은 **사후 수습**이고,
근본 대응은 **마스크 조건부 추정**이다. 이는 백엔드 선택에 직접 영향을 준다(§4의 후보 C3 참조).

---

## 3. 후보 (배포처·라이선스 확정)

라이선스 상세는 [licenses.md](../licenses.md)에 매트릭스로 관리한다.

### 티어 A — 경량 2D (실시간 v3, 빠른 반복)

| 후보 | HF repo | 라이선스 | 판단 |
| --- | --- | --- | --- |
| **RTMPose (Halpe-26) ONNX** | `pocketpose/onnx-models` (`pose_estimation-halpe26`) | Apache-2.0 | **1순위.** 발 관절 보유(C1 충족), ONNX Runtime 단독 실행 → mmcv 컴파일 불필요(R8 회피) |
| ViTPose (COCO-17) | `usyd-community/vitpose-plus-base` | Apache-2.0 | transformers 네이티브로 가장 간단하나 **발 관절 없음** → M1 스캐폴드 검증용으로만 |
| MediaPipe Pose | `qualcomm/MediaPipe-Pose-Estimation` (미러) | Apache-2.0 | BlazePose-33. 발 관절 있음. 정확도 하위, 폴백 후보 |

> M1(§13-4번)은 `usyd-community/vitpose-plus-base`로 시작해도 된다. 목표가 "스켈레톤 렌더 확인"이므로
> 발 관절이 필요 없고, transformers 하나로 끝나 Blackwell 환경 검증(R8)을 가장 빨리 통과시킨다.
> **단 M3(AXI 지표) 진입 전에 Halpe-26으로 교체해야 한다.**

### 티어 B — 2D + 3D 리프팅

| 후보 | 배포처 | 라이선스 | 판단 |
| --- | --- | --- | --- |
| RTMPose/ViTPose + MotionBERT 계열 | GitHub, HF 공식 없음 | 확인 필요 | C2로 인해 **우선순위 하향.** 벤치마크 대조군으로만 유지 |

### 티어 C — World-grounded

| 후보 | 배포처 | 라이선스 | 판단 |
| --- | --- | --- | --- |
| **GVHMR** | GitHub(zju3dv) / `ryanrudes/gvhmr`(커뮤니티 미러) | SMPL 종속 → **비상업** | **1순위.** Gravity-View 좌표계로 중력 정렬 좌표를 직접 출력 → §5.2 난제를 모델이 해소 |
| **NLF** | GitHub(isarandi) — 코드 MIT, **가중치 비상업 연구용** | 비상업 | **2순위.** 정준 인체 볼륨의 임의 점을 질의 가능 → DVI-Canonical 관절을 어댑터 매핑 없이 직접 획득. 다인·메트릭 스케일·빠름. 단 카메라 좌표계 출력 |
| **PromptHMR (-Vid)** | GitHub | SMPL 종속 → 비상업 | **C3 대응 후보.** 세그멘테이션 마스크 + 밀접접촉 라벨을 프롬프트로 받음 → 아브라소와 설계 가정이 일치. world-grounded 정확도 최상위권. 대가는 무게(SLAM + 깊이 추정) → NFR-01 위협 |
| Sapiens / Sapiens2 (pose) | `facebook/sapiens2-pose-1b`, `facebook/sapiens-pose-1b` | v1 CC-BY-NC / v2 Sapiens2 License(**조항 확인 필요**) | 308 keypoint·1K 해상도로 정밀도 최상. 1.5B 파라미터 top-down은 NFR-01에 부담. 발/전신 관절 확보 목적의 예비 후보 |

### GVHMR을 1순위로 두는 이유

ARCHITECTURE §5.2는 중력 정렬을 "이 프로젝트의 핵심 난제"로 규정하고 4단 폴백 체인을 설계했다.
GVHMR은 중력 방향과 카메라 시선으로 정의된 좌표계에서 프레임별 포즈를 예측하고 중력 제약으로 합성하므로,
`provides_world_frame = True`가 된다. 즉:

- §5.2 폴백 1순위("모델 제공")가 실제로 성립 → **2·3순위(바닥 RANSAC, 장축 통계)를 v1에서 구현하지 않아도 된다**
- R3(카메라 틸트) 상당 부분 해소, R2(단안 깊이 모호성)도 완화
- §13-8번(`geometry/gravity.py` 폴백 체인) 작업 범위가 크게 줄어든다 — 4순위 수동 캘리브레이션만 폴백으로 남긴다

**주의**: GVHMR은 "모델 하나"가 아니라 파이프라인이다(내부적으로 2D 포즈 + SLAM 사용). 공식 데모는 1인 기준이므로
**커플은 트랙별로 분리 실행**해야 하고, 이는 처리 시간이 인물 수에 선형 비례함을 뜻한다(NFR-01 계산에 반영).

---

## 4. 부수 결정 (같은 조사에서 파생)

이 ADR의 범위는 아니지만 백엔드 선택과 맞물리므로 기록한다. 각각 별도 ADR 대상.

| # | 항목 | 제안 | 근거 |
| --- | --- | --- | --- |
| C4-1 | **인물 검출** | `PekingU/rtdetr_v2_r50vd` (Apache-2.0) | transformers 네이티브, NMS 불필요. **Ultralytics YOLO는 AGPL-3.0이므로 R7 관점에서 회피** |
| C4-2 | **ReID 임베딩** | `facebook/dinov2-base` (Apache-2.0) | §5.3의 외형 임베딩 층. DINOv3는 자체 라이선스라 v2가 안전 |
| C4-3 | **가림 대응 마스크** | `facebook/sam2.1-hiera-large` (Apache-2.0) | 비디오 마스크 전파로 겹침 중에도 인물별 마스크 유지 → ① R6 ID 스와핑 방어의 4번째 층, ② top-down 크롭 오염 방지(C3), ③ PromptHMR 전환 시 필요한 입력 선확보 |
| C4-4 | **코칭 문장 (FR-08)** | `Qwen/Qwen3-8B` 또는 `google/gemma-3-12b-it` | §7.3 원칙대로 **수치·규칙이 원본, LLM은 표현만**. 로컬 실행으로 NFR-05 유지 |

---

## 5. M2 선정 절차 (ARCHITECTURE §5.1 구체화)

1. **검증 클립 준비** — 동일 동작을 정면/45°/측면 3각도 **동시 촬영** (PROJECT.md V2). 오픈 엠브레이스 1클립 + 클로즈 엠브레이스 1클립.
2. **후보 3개만 비교** — (B) 2D+MotionBERT, (C1) GVHMR, (C2) NLF. PromptHMR은 C3가 실제로 문제가 될 때 4번째로 투입.
3. **평가 축과 가중치**

   | 축 | 측정 | 가중 |
   | --- | --- | --- |
   | **3각도 간 일관성** (주) | POS-01/POS-02를 3각도에서 산출 → 표준편차 | 최우선 (NFR-06, SC-4) |
   | 육안 타당성 | 오버레이에서 스켈레톤이 몸을 따라가는가 | 게이트 (실패 시 탈락) |
   | 클로즈 엠브레이스 커버리지 | `valid` 비율 (§4.5 coverage) | 높음 (R1) |
   | 처리 속도 | 3분/1080p/30fps, 2인 기준 실측 | NFR-01 게이트 (10분) |
   | 라이선스 | licenses.md 매트릭스 | 기록만 (v1은 로컬 개인 도구) |
   | 가중치 조달 재현성 | 공식 배포 여부, 해시 고정 가능성 | 중간 (NFR-03) |

4. **판정 결과를 이 문서의 §6에 추가하고 상태를 Accepted로 변경**한다.

> 주의: 이 벤치마크는 "절대 정확도"를 측정하지 않는다. 정답 라벨이 없다(R4).
> 측정하는 것은 **일관성**이다 — 같은 동작을 다른 각도에서 찍었을 때 지표가 같은 값을 내는가.

---

## 6. 결정 (M2 이후 작성)

*미정. 벤치마크 수행 후 채운다.*

---

## 7. 결과 및 후속 작업

### 채택 시 영향 (GVHMR 가정)

- `geometry/gravity.py`: 폴백 체인 4단 → **모델 제공 + 수동 캘리브레이션 2단으로 축소**
- `perception/pose/tier_c_gvhmr.py`: `provides_world_frame = True` → GravityAlign 스테이지 스킵 경로 필요
- 처리 시간이 인물 수에 선형 → NFR-01 재계산 및 `configs/backends/tier_c.yaml`에 인물 수 상한 명시
- SMPL 바디 모델 별도 등록·다운로드 절차를 `scripts/download_models.py`에 수동 단계로 문서화

### 미해결

| # | 질문 | 시점 |
| --- | --- | --- |
| D1 | Sapiens2 License의 상업 이용 조항 원문 확인 | 후보로 승격될 때 |
| D2 | 커뮤니티 미러(`ryanrudes/gvhmr`) 의존을 허용할지, 공식 GitHub 릴리스만 쓸지 | M2 |
| D3 | Halpe-26 기반 DVI-Canonical 확정 (A1 종결) | M1 종료 시 |
| D4 | 커플 2인을 트랙별 분리 실행 vs 다인 동시 처리 모델(NLF) | M4 |

---

## 8. 조사 출처

- [usyd-community/vitpose-plus-base](https://huggingface.co/usyd-community/vitpose-plus-base) · [ViTPose in transformers](https://huggingface.co/docs/transformers/en/model_doc/vitpose)
- [pocketpose/onnx-models (Halpe-26)](https://huggingface.co/pocketpose/onnx-models/tree/main/pose_estimation-halpe26) · [RTMPose (mmpose)](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)
- [GVHMR (zju3dv)](https://github.com/zju3dv/GVHMR) · [ryanrudes/gvhmr](https://huggingface.co/ryanrudes/gvhmr) · [논문](https://huggingface.co/papers/2409.06662)
- [NLF (isarandi)](https://github.com/isarandi/nlf) · [nlf-pipeline](https://github.com/isarandi/nlf-pipeline)
- [PromptHMR](https://arxiv.org/html/2504.06397)
- [facebook/sapiens2-pose-1b](https://huggingface.co/facebook/sapiens2-pose-1b) · [facebookresearch/sapiens2](https://github.com/facebookresearch/sapiens2)
- [DINOv3 in transformers](https://huggingface.co/docs/transformers/model_doc/dinov3)
