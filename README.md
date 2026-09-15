# DanceVideoInstructor (dvi)

아르헨티나 탱고 영상에서 포즈 스켈레톤을 추출해 자세(상체 정렬)와 축(axis/balance)을
정량화하고, 문제 구간을 근거 프레임과 함께 제시하는 분석 도구.

## 문서 지도

| 문서 | 내용 |
| --- | --- |
| [docs/PROJECT.md](docs/PROJECT.md) | 프로젝트 정의 — 배경, 목표/비목표, 요구사항, 검증 전략, 리스크 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 아키텍처 — 파이프라인 스테이지, 데이터 계약, **지표 카탈로그**(§6), 구현 순서 |
| [docs/PROGRESS.md](docs/PROGRESS.md) | 진행 상황 스냅샷 — 뭐가 되고 뭐가 안 되는지, 발견된 버그, 다음 작업 |
| [docs/EVALUATION.md](docs/EVALUATION.md) | 평가 노트 — 지금 바로 뽑을 수 있는 정량 지표, 오픈 데이터셋으로 컴포넌트 검증하기 |
| [docs/adr/0001-pose-backend.md](docs/adr/0001-pose-backend.md) | ADR — 포즈 백엔드 선정 (제안 상태, M2 벤치마크 이후 확정) |
| [docs/licenses.md](docs/licenses.md) | 모델/데이터셋 라이선스 매트릭스 |
| [samples/README.md](samples/README.md) | 실행 가능한 예제 — GPU/영상 없이 지표 엔진 돌려보기 |

---

## 상태

설계는 끝났고(`docs/PROJECT.md`, `docs/ARCHITECTURE.md`), 구현은 **M1(1인 스켈레톤) 완료,
M2(중력 정렬)·M3(자세·축 지표)은 핵심 로직 구현·검증 완료 + 파이프라인 미연결** 상태다.
`dvi analyze` 같은 end-to-end CLI는 아직 없다 — 모든 지표는 함수 단위로만 테스트됐다.

| 단계 | 상태 |
| --- | --- |
| M0. 스캐폴드 | ✅ 완료 |
| M1. 1인 스켈레톤 | ✅ 완료 — 실사 영상 2개로 검증 |
| M2. 3D + 중력 정렬 | 🟡 부분 — 중력 정렬 폴백체인 구현·합성검증 완료. 포즈 백엔드 자체는 미확정(ADR-0001 Proposed) |
| M3. 자세·축 지표 | 🟡 부분 — POS 6개 · AXI 4개 지표 함수 합성 검증 완료. 실제 파이프라인/CLI 미연결 |
| M4. 커플 확장 | 🟡 착수 — RT-DETRv2 인물 검출기(`perception/detect/rtdetr.py`) 구현 + 실험 스크립트(`scripts/two_person_axis.py`)로 시험 중. 프레임 간 ID 추적·역할 분류는 아직 없음 |
| M5~M8 | ⬜ 미착수 |

자세한 마일스톤별 근거, 발견된 실제 버그, "다음으로 자연스러운 작업"은
[docs/PROGRESS.md](docs/PROGRESS.md)를 참고 — 이 README보다 그쪽이 항상 최신이다.

---

## 개발 환경

```bash
uv sync
uv run dvi check-env
uv run dvi --help
```

- Python 3.12 고정 (`uv python pin 3.12`)
- GPU는 RTX 50xx(Blackwell) 기준 **CUDA 12.8+** 빌드 필요 (`pyproject.toml`의
  `pytorch-cu128` 인덱스 참고). `uv run dvi check-env`가 torch/CUDA 연결을 확인해준다
- CPU만으로도 아래 "지금 실행 가능한 것들" 중 `samples/metrics_demo.py`와
  `pytest`는 돌아간다 — 모델 추론이 필요한 것만 GPU를 권장

---

## 지금 실행 가능한 것들

`dvi analyze`(전체 파이프라인 CLI)는 아직 스텁이다 (`uv run dvi analyze ...`를 실행하면
"not implemented yet"이 뜬다). 대신 아래가 실제로 동작한다:

```bash
# 1. 환경 확인 (Python/CUDA/torch)
uv run dvi check-env

# 2. 지표 엔진 전체를 GPU/영상 없이 바로 확인 (샘플 시퀀스 + 카메라 일관성 하네스)
uv run python samples/metrics_demo.py

# 3. 실제 영상 1개로 1인 스켈레톤 추론 + 오버레이 렌더 (ViTPose, 첫 실행 시 가중치 다운로드)
uv run python scripts/tier_a_smoke.py data/raw/sample1_front.mp4

# 4. 커플 영상에서 RT-DETRv2 검출 + 인물별 포즈 실험 (프레임 간 ID 추적은 아직 없음)
uv run python scripts/two_person_axis.py data/raw/sample1_front.mp4

# 5. 테스트 스위트 (126 passed, 수 초 내 완료)
uv run pytest -q
```

`data/raw/*.mp4`는 저장소에 포함된 실제 탱고 연습 영상 샘플이다
(`docs/PROGRESS.md` §4.1 참고). 각 스크립트가 하는 일과 출력 예시, 함정은
[samples/README.md](samples/README.md)에 자세히 정리했다.

---

## 지표 카탈로그 (구현된 것만 — 전체 설계는 `docs/ARCHITECTURE.md` §6)

| 카테고리 | 지표 | 구현 파일 | 검증 상태 |
| --- | --- | --- | --- |
| **자세 (POS)** | POS-01 상체 전후 기울기, POS-02 좌우 기울기, POS-03 척추 꺾임, POS-04 머리 전방 이동, POS-05a 어깨 거상, POS-05b 어깨 비대칭, POS-06 상체 직선성 | [src/dvi/metrics/posture.py](src/dvi/metrics/posture.py) | 합성 기하 검증(단위 테스트) 완료. 실사 영상에도 적용 가능 |
| **축/균형 (AXI)** | AXI-01 무게중심-지지발 오프셋, AXI-02 축 흔들림(저크), AXI-04 지지 무릎 굴곡, AXI-06 축 수직성 | [src/dvi/metrics/axis.py](src/dvi/metrics/axis.py) | 합성 기하 검증 완료. **실사 영상 적용은 Tier A(COCO-17)에 발 관절이 없어 제한적** (ADR-0001 §2 C1) |
| 축/균형 (AXI), 미구현 | AXI-03 체중이동 명료성, AXI-05 접지 품질 | — | Segment 스테이지(이벤트 검출) 필요, 미착수 |
| 회전/분리 (DIS), 커플 (CPL) | 전체 | — | 다중 인물 추적(M4) 선행 필요, 미착수 |

지지발 판정(`src/dvi/geometry/support_foot.py`)과 무게중심 추정(`src/dvi/geometry/com.py`)이
AXI-\* 전체의 기반이다. `POS-01`(전신이 하나로 기운 것, 스타일에 따라 정상)과
`POS-03`(허리에서 꺾이는 것, 항상 문제)을 분리 측정하는 게 설계의 핵심 — 자세한 이유는
`src/dvi/metrics/posture.py` 모듈 docstring 참고.

---

## 평가/검증

정답 라벨이 없는 도메인이라(`PROJECT.md` R4) "그래프가 절대적으로 맞는가"는 아직 답할 수
없다. 대신 다층 검증(`PROJECT.md` §7)을 쌓는다:

- **V1 합성/기하 검증** — 지금 바로 가능, 구현된 11개 지표 전부 적용됨 (`tests/unit/test_metrics_*.py`)
- **V2 카메라 각도 일관성** — 하네스(`eval/harness.py`)는 구현·검증됐지만, 동시 다각도 촬영 영상이 아직 없어 실측은 못 함
- **V3 대조군 테이크** / **V4 전문가 라벨** / **V5 임계값 캘리브레이션** — 미착수

**"지금 이 커밋에서 라벨 없이 뽑을 수 있는 정량 지표가 더 있는가"**와 **"탱고 데이터가
없는데 어떤 오픈 데이터셋으로 개별 컴포넌트(검출/포즈/중력정렬/추적)를 미리 검증할 수
있는가"**는 [docs/EVALUATION.md](docs/EVALUATION.md)에 정리했다 — Halpe/COCO-WholeBody
(발 관절 정확도), AIST++(다각도 동기화 촬영 대체재), CHI3D/Hi4D(클로즈 엠브레이스 대체재),
CrowdHuman/MOT 계열(검출·추적 벤치마크) 등 후보와 권장 순서를 담고 있다.

---

## 리포지토리 구조

```
src/dvi/
├─ cli.py                    골격만 (check-env만 실동작, 나머지는 마일스톤 명시된 스텁)
├─ schemas/                  Frame, PoseSequence, MetricResult, Finding, AnalysisReport ...
├─ pipeline/                 Stage 프로토콜, ArtifactStore 캐시, BatchRunner
├─ io/                       video_reader, video_writer
├─ perception/
│  ├─ pose/                  base(Pose2D), skeleton(COCO17 매핑), tier_a_vitpose
│  ├─ detect/                RT-DETRv2 인물 검출기 (파이프라인 미통합, 단위 테스트 없음)
│  └─ track/                 빈 패키지 (M4)
├─ geometry/                 gravity, calibration, com, support_foot, normalize
├─ metrics/                  posture(7개), axis(4개) — dissociation/couple/rhythm 없음
├─ segment/                  빈 패키지
├─ scoring/                  빈 패키지
├─ report/                   overlay만 (json_report, html 없음)
└─ reference/                빈 패키지 (M7)

eval/harness.py               angle_consistency만 (라벨 상관 분석은 라벨 없어서 미착수)
scripts/                      check_env, tier_a_smoke(1인 스모크), two_person_axis(커플 실험)
samples/                      GPU/영상 없이 돌려보는 튜토리얼 (samples/README.md)
configs/                      백엔드 티어별, 스타일 프로파일별(살롱/밀롱게로/누에보) 설정
tests/unit, tests/integration 합성 픽스처 기반 단위 테스트 + 통합 테스트
```

`docs/PROGRESS.md` §5가 더 세밀한 버전을 유지한다.
