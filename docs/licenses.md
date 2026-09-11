# 모델 라이선스 매트릭스

> PROJECT.md R7("모델 라이선스 — SMPL 계열은 비상업 제약 있음") 대응 문서.
> 백엔드를 교체 가능하게 만든 이유의 절반은 라이선스다. 후보를 추가할 때마다 이 표를 갱신한다.

| 항목 | 값 |
| --- | --- |
| 문서 버전 | v0.1 (초안) |
| 조사 시점 | 2026-09-11 |
| 관련 문서 | [adr/0001-pose-backend.md](./adr/0001-pose-backend.md), [ARCHITECTURE.md](./ARCHITECTURE.md) §5.1 |

---

## 0. 전제: v1의 이용 형태

v1은 **단일 사용자 로컬 도구**이고 전 과정 로컬 처리(NFR-05), 외부 배포 없음이다.
따라서 **비상업 제약이 있는 모델도 v1에서는 사용 가능**하다.
이 표의 실질적 용도는 두 가지다.

1. 나중에 배포·상업화를 고려할 때 **무엇을 갈아끼워야 하는지** 미리 아는 것
2. 가중치를 받으려면 **별도 등록/동의 절차**가 필요한 모델을 놓치지 않는 것 (`scripts/download_models.py`)

---

## 1. 요약 — 등급 구분

| 등급 | 의미 | 해당 |
| --- | --- | --- |
| 🟢 **자유** | 상업 이용 가능, 표시 의무 정도 | Apache-2.0, MIT, BSD |
| 🟡 **조건부** | 상업 가능하나 조항 확인/준수 필요 | Gemma Terms, Sapiens2 License(미확인), 자체 라이선스 |
| 🔴 **비상업** | 연구·개인 용도로 제한 | CC-BY-NC, SMPL 계열, NLF 가중치 |
| ⚫ **전염성** | 사용 시 파생물 공개 의무 | AGPL-3.0 |

---

## 2. 포즈·인체 복원

| 모델 | 배포처 | 코드 | 가중치 | 등급 | 비고 |
| --- | --- | --- | --- | --- | --- |
| **ViTPose / ViTPose++** | `usyd-community/vitpose-*` (HF) | Apache-2.0 | Apache-2.0 | 🟢 | transformers 네이티브. COCO-17 → 발 관절 없음 |
| **RTMPose / RTMW** | `pocketpose/onnx-models`, `qualcomm/RTMPose-Body2d` (HF), open-mmlab/mmpose | Apache-2.0 | Apache-2.0 | 🟢 | Halpe-26 / WholeBody-133 포맷 선택 가능 |
| **MediaPipe Pose** | google, `qualcomm/MediaPipe-Pose-Estimation` (HF 미러) | Apache-2.0 | Apache-2.0 | 🟢 | BlazePose-33 |
| **GVHMR** | zju3dv (GitHub), `ryanrudes/gvhmr` (HF 커뮤니티 미러) | 리포 LICENSE 확인 필요 | **SMPL 종속** | 🔴 | §3의 SMPL 제약이 그대로 전이 |
| **NLF** | isarandi (GitHub) | MIT | **비상업 연구용** (릴리스 명시) | 🔴 | 코드와 가중치의 라이선스가 다름 — 가중치가 구속 조건 |
| **PromptHMR** | GitHub | 확인 필요 | **SMPL 종속** | 🔴 | 마스크 조건부 추정. 클로즈 엠브레이스 대응 후보 |
| **TRAM** | GitHub | 확인 필요 | **SMPL 종속** | 🔴 | + DROID-SLAM, ZoeDepth 라이선스도 함께 확인 필요 |
| **MotionBERT 계열** | GitHub | 확인 필요 | 확인 필요 | 🟡 | HF 공식 배포 없음 (ADR-0001 §2 C2) |
| **Sapiens (v1) pose/depth** | `facebook/sapiens-pose-1b` 등 (HF) | — | **CC-BY-NC 4.0** | 🔴 | |
| **Sapiens2 pose** | `facebook/sapiens2-pose-1b` 등 (HF) | — | **Sapiens2 License** | 🟡 | **D1: 상업 조항 원문 미확인.** 후보 승격 시 확인 |

### SMPL / SMPL-X 바디 모델 — 가장 주의할 항목

| 항목 | 내용 |
| --- | --- |
| 무엇 | SMPL·SMPL-X **바디 모델 파라미터 파일** (`.npz`/`.pkl`). 모델 가중치와 **별개**로 필요 |
| 어디서 | `smpl-x.is.tue.mpg.de` 등에서 **계정 등록 + 라이선스 동의 후 수동 다운로드** |
| 제약 | **비상업 연구 목적 한정** |
| 영향 | Tier C 후보(GVHMR, PromptHMR, TRAM, SMPLer-X 계열) **전부** 이 제약을 상속 |
| 대응 | `scripts/download_models.py`는 이 파일을 자동 다운로드할 수 없다 → **수동 단계로 문서화**하고 존재 여부만 검증 |

> **상업화 시나리오가 생기면 Tier C 전체를 버려야 한다.** 그때의 대안은 SMPL을 쓰지 않는
> 스켈레톤 기반 경로(RTMPose + 자체 리프팅) 또는 SMPL-free 모델이다.
> 어댑터(§5.1 `PoseBackend` 프로토콜)를 지키는 것이 이 리스크의 유일한 보험이다.

---

## 3. 검출·추적·분할

| 모델 | 배포처 | 라이선스 | 등급 | 비고 |
| --- | --- | --- | --- | --- |
| **RT-DETRv2** | `PekingU/rtdetr_v2_r50vd` (HF) | Apache-2.0 | 🟢 | **검출 1순위.** transformers 네이티브 |
| **SAM 2.1** | `facebook/sam2.1-hiera-large` (HF) | Apache-2.0 | 🟢 | 비디오 마스크 전파. R1/R6 대응 |
| **DINOv2** | `facebook/dinov2-base` (HF) | Apache-2.0 | 🟢 | **ReID 임베딩 1순위** |
| **DINOv3** | `facebook/dinov3-*` (HF) | DINOv3 License | 🟡 | 성능 우위이나 자체 라이선스 → v1은 v2 사용 |
| **Ultralytics YOLO (v8~v11, pose 포함)** | HF, ultralytics | **AGPL-3.0** | ⚫ | **회피.** 네트워크 서비스로 제공 시 전체 소스 공개 의무 → FastAPI(v2)·뷰어 계획과 충돌 |
| ByteTrack / BoT-SORT | GitHub (알고리즘) | 원 리포 확인 필요 | 🟡 | 모델이 아님. 구현 이식 시 원 리포 라이선스 확인 |

> **AGPL 주의**: Ultralytics는 편의성 때문에 유혹적이지만, ARCHITECTURE §2의 FastAPI 진입점과 결합되면
> "네트워크를 통한 이용" 조항에 걸린다. 개인 로컬 사용만 하면 문제없으나, **처음부터 안 쓰는 쪽이 싸다.**

---

## 4. LLM (FR-08 코칭 문장, 선택)

| 모델 | 배포처 | 라이선스 | 등급 |
| --- | --- | --- | --- |
| **Qwen3-8B / 14B** | `Qwen/Qwen3-8B` (HF) | Apache-2.0 | 🟢 |
| **Gemma 3 12B IT** | `google/gemma-3-12b-it` (HF) | Gemma Terms of Use | 🟡 |

> §7.3 원칙: **수치와 규칙이 원본, LLM은 표현만.** LLM 미사용(템플릿 전용)이 v1 기본값이므로
> 이 항목은 선택 의존성이며 라이선스 리스크도 선택적이다.

---

## 5. 데이터셋 (간접 제약)

모델 가중치가 학습된 데이터셋의 제약이 가중치로 전이되는 경우가 있다. Tier C 후보 대부분은
AMASS / BEDLAM / Human3.6M / 3DPW 계열로 학습되며, **Human3.6M은 비상업 연구 한정**이다.
이것이 Tier C 가중치의 비상업 성격을 강화하는 두 번째 이유다.

또한 `data/reference/`(§7.4 레퍼런스 라이브러리)에 넣는 **탱고 영상 자체의 사용 권한**을
클립별 메타에 기록해야 한다. 공연 영상은 저작권이 있다.

---

## 6. 후속 작업

| # | 항목 | 시점 |
| --- | --- | --- |
| L1 | Sapiens2 License 원문 확인 (ADR-0001 D1) | 후보 승격 시 |
| L2 | GVHMR·PromptHMR·TRAM 리포 LICENSE 파일 원문 확인 | M2 벤치마크 시 |
| L3 | 선정된 백엔드의 전체 의존 그래프 라이선스 감사 (SLAM·깊이 추정 등 부속 모델 포함) | M2 종료 시 |
| L4 | `scripts/download_models.py`에 수동 동의 필요 항목 게이트 구현 | M2 |

> 이 문서는 법률 자문이 아니다. 각 항목은 배포처의 라이선스 원문을 근거로 갱신하며,
> **"확인 필요"로 표시된 칸을 근거 없이 채우지 않는다.**
