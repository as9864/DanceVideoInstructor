# DanceVideoInstructor — 진행 상황

> 이 문서는 스냅샷이다. "지금 뭐가 되고 뭐가 안 되는지"를 담는다.
> 배경/범위는 [PROJECT.md](./PROJECT.md), 설계는 [ARCHITECTURE.md](./ARCHITECTURE.md) 참고.

| 항목 | 값 |
| --- | --- |
| 문서 버전 | v0.1 |
| 작성일 | 2026-09-13 |
| 대상 커밋 | `f8784ec` (마지막) ~ `f8a5fbc` (첫) |
| 테스트 현황 | 126 passed (`uv run pytest -q`) |

---

## 1. 한 줄 요약

M0(스캐폴드)~M1(단일 인물 스켈레톤)은 완료, M2(중력 정렬)·M3(자세·축 지표)은 **핵심 로직은 구현·검증됐지만 실제 파이프라인으로 연결되지 않은 상태**다. 실제 탱고 영상 2개로 검증한 결과, 현재 구조의 근본적 다음 병목은 예상대로 **다중 인물 검출/추적**(M4)이라는 게 확인됐다.

---

## 2. 마일스톤 현황 (PROJECT.md §6.2 기준)

| 단계 | 완료 조건 | 상태 |
| --- | --- | --- |
| **M0. 스캐폴드** | `dvi --help` 동작, CI 그린 | ✅ 완료 (CI 파이프라인 자체는 미구성 — 로컬 테스트만) |
| **M1. 1인 스켈레톤** | 단일 인물 영상에서 스켈레톤 렌더 | ✅ 완료 — 실사 영상으로 검증 (§5) |
| **M2. 3D + 중력 정렬** | 카메라 각도 달라도 수직축 일치 | 🟡 부분 — 중력 정렬 폴백체인은 구현·합성검증 완료. **포즈 백엔드 자체가 미확정**(ADR-0001 Proposed), 3D 리프팅 없음 |
| **M3. 자세·축 지표** | 지표 그래프 산출, 눈으로 검증 | 🟡 부분 — POS 6개·AXI 4개 지표 함수는 합성 데이터로 검증 완료. **실제 파이프라인/CLI에 연결 안 됨**, 그래프 시각화 없음 |
| **M4. 커플 확장** | 다중 추적, 역할 분류, CPL-* 지표 | ⬜ 미착수 — `perception/detect/`, `perception/track/`는 빈 패키지 |
| **M5. 채점 + 리포트** | 루브릭, 문제 검출, JSON/HTML 리포트 | ⬜ 미착수 |
| **M6~M8** | 평가 하네스 / 레퍼런스 비교 / 실시간 | ⬜ 미착수 (M6은 일관성 하네스만 부분 준비, §4 참고) |

---

## 3. ARCHITECTURE.md §13 구현 순서별 상세

| # | 작업 | 상태 | 커밋 |
| --- | --- | --- | --- |
| 1 | `scripts/check_env.py` | ✅ | `f8a5fbc` |
| 2 | `schemas/` 전체 + 좌표계 정의 | ✅ | `da2bc2c` |
| 3 | `io/video_reader.py` + `pipeline/` + 캐시 | ✅ | `ce2b58c` |
| 4 | Tier A 백엔드 + `skeleton.py` 매핑 + 오버레이 → **M1** | ✅ | `6516b17`, `4f12ffd` |
| 5 | `tests/fixtures/synthetic/` 합성 스켈레톤 생성기 | ✅ | `6516b17` |
| 6 | `metrics/posture.py` POS-01/03 + 단위테스트 | ✅ | `6516b17` |
| 7 | Tier B/C 백엔드 벤치마크 → ADR-0001 확정 → **M2** | 🟡 인프라만 | `ef6e08b` (아래 §4.2 참고) |
| 8 | `geometry/gravity.py` 폴백 체인 | ✅ | `ef6e08b` |
| 9 | 나머지 POS-*, AXI-* + 지지발 상태기계 → **M3** | 🟡 대부분 | `f8784ec` (AXI-03/05 제외, 아래 §4.3 참고) |
| 10 | 추적·역할 분류·CPL-* → **M4** | ⬜ | — |
| 11 | 루브릭·Finding·리포트 → **M5** | ⬜ | — |

### 4.1 — M1: 실사 영상 검증 (2026-09-11)

사용자가 제공한 실제 탱고 연습 영상 2개(`sample1_front.mp4` 9.7분, `sample2_front.mp4` 9.1분, 1280x720@30fps)로 전체 파이프라인(ViTPose 추론 → DVI-Canonical 매핑 → 오버레이 렌더)을 끝까지 돌렸다.

| 항목 | sample1 | sample2 |
| --- | --- | --- |
| 추론 속도 | 49.2 fps | 50.9 fps |
| 평균 신뢰도 | 0.529 | 0.551 |
| 커버리지 | 89.0% | 93.7% |

**속도는 NFR-01(3분/1080p를 10분 이내) 목표를 여유 있게 만족** — RTX 5070 Ti + cu128 빌드가 실제로 작동함을 확인.

**중요 발견**: 두 영상 모두 걷기(캐미나타) 구간은 정확히 추적되지만, 클로즈 엠브레이스로 들어가면 두 사람이 겹치면서 스켈레톤이 두 인물 사이에서 혼동되는 현상을 확인했다. 이는 PROJECT.md R1과 ADR-0001 §2 C3이 이미 예견한 한계이며, 근본 원인은 **아직 실제 인물 검출기(Detect 스테이지)가 없어서** 화면 전체를 박스 하나로 취급하기 때문이다. 버그가 아니라 M1 범위("단일 인물")의 경계를 실측으로 확인한 것 — M4(추적) 필요성을 앞당겨 확인시켜 준 결과.

렌더링된 오버레이(`data/outputs/*.mp4`, 각 ~260~285MB)는 로컬에만 존재하고 `.gitignore`로 제외됨.

### 4.2 — M2: Tier B/C 벤치마크가 완료되지 못한 이유

ADR-0001 §5가 요구하는 실제 절차(정면/45°/측면 동시촬영 + GVHMR)는 이번 세션에서 완료 불가능했다:

1. **동시 다각도 촬영 영상 없음** — 세션 중 확보한 영상은 정면 1대뿐
2. **GVHMR은 SMPL 바디 모델이 필요**하고, 이는 `smpl-x.is.tue.mpg.de`에서 사람이 직접 계정 등록·동의해야 받을 수 있음 (자동화 불가)

대신 실제 데이터/백엔드가 준비되면 바로 쓸 수 있는 **[eval/harness.py](../eval/harness.py)**(`angle_consistency` — 여러 각도의 `MetricResult`를 받아 프레임별 교차 각도 표준편차를 계산)를 미리 구현·검증해 두었다.

### 4.3 — M3: 구현된 지표와 제외된 지표

**구현·검증됨** (합성 스켈레톤 기하 검증, PROJECT.md §7 V1 방식):

| 지표 | 파일 | 비고 |
| --- | --- | --- |
| POS-01 | [metrics/posture.py](../src/dvi/metrics/posture.py) | 상체 전후 기울기 |
| POS-02 | 〃 | 상체 좌우 기울기 |
| POS-03 | 〃 | 척추 꺾임 (POS-01과 수학적으로 독립임을 테스트로 확인) |
| POS-04 | 〃 | 머리 전방 이동 (BODY 프레임) |
| POS-05a | 〃 | 어깨 거상 — **문서의 "p10" 그대로 쓰면 신호가 뒤집혀서 p90으로 수정**(코드 주석에 이유 명시) |
| POS-05b | 〃 | 어깨 비대칭 |
| POS-06 | 〃 | 상체 직선성 (4점 직선적합, 벡터화) |
| AXI-01 | [metrics/axis.py](../src/dvi/metrics/axis.py) | 무게중심-지지발 오프셋 |
| AXI-02 | 〃 | 축 흔들림 (프레임별 저크 크기로 단순화 — 원래 스펙의 "구간별 표준편차"와 다름, 문서화됨) |
| AXI-04 | 〃 | 지지 무릎 굴곡 |
| AXI-06 | 〃 | 축 수직성 |

**제외됨** (이번 세션 범위 밖, 이유 명시):

| 지표 | 제외 이유 |
| --- | --- |
| AXI-03 (체중이동 명료성) | 이벤트/구간 검출(Segment 스테이지, 파이프라인 스테이지 11) 필요 — 아직 미구현 |
| AXI-05 (접지 품질) | 착지 순간의 발끝→뒤꿈치 순서 검출 필요, AXI-03과 같은 이유로 보류. 또한 Tier A(COCO-17)는 발 관절 자체가 없어 실제 백엔드로는 애초에 측정 불가 (ADR-0001 C1) |
| CPL-* (커플 지표 전체) | 다중 인물 추적(M4) 선행 필요 |

**기반 모듈**: [geometry/com.py](../src/dvi/geometry/com.py)(무게중심), [geometry/support_foot.py](../src/dvi/geometry/support_foot.py)(지지발 히스테리시스 상태기계), [geometry/normalize.py](../src/dvi/geometry/normalize.py)(BODY 프레임).

**중요한 제약**: 이 지표들은 전부 **합성 데이터로만 검증**되었다. 실제 영상 기반 검증은 Tier A(COCO-17)에 발 관절이 없어 AXI-01/06 같은 발 의존 지표는 진짜 영상에 적용 불가능하고, M3을 완결하려면 ADR-0001의 Halpe-26급 백엔드 교체가 선행돼야 한다.

---

## 4. 발견된 실제 버그 (수정 완료)

작업 중 발견해서 고친, 문서에는 없던 실제 결함들:

| # | 버그 | 원인 | 수정 |
| --- | --- | --- | --- |
| 1 | 리사이즈 시 홀수 해상도로 인한 `libx264` 인코딩 크래시 | `yuv420p`는 짝수 폭/높이 필요, 반올림이 홀수를 만들 수 있었음 | `VideoReader._scaled_size`가 항상 짝수로 반올림 |
| 2 | `render_overlay`가 추론 해상도와 다른 해상도로 영상을 다시 읽으면 스켈레톤이 몸에서 벗어남 | `pose.image_size`와 실제 렌더 해상도 불일치를 보정하지 않음 | `pose.image_size` 기준 자동 리스케일 추가 + 회귀 테스트 |
| 3 | Windows 콘솔(cp949)에서 em-dash 등 non-ASCII 문자 출력 시 크래시 | 코리안 로케일 콘솔 코드페이지가 UTF-8이 아님 | `cli.py`/`check_env.py`에서 stdout/stderr UTF-8 강제 재설정 |
| 4 | `models/.gitkeep`이 `.gitignore`의 `models/` 규칙 때문에 추적 안 됨 | 디렉터리 전체가 무시되면 그 안의 `!예외` 규칙이 작동하지 않는 git의 알려진 동작 | `models/*` + `!models/.gitkeep`으로 변경 |
| 5 | POS-05a에서 스펙의 "p10" 그대로 구현하면 어깨가 이완될 때 값이 커지고 으쓱할 때 작아짐 (신호 반전) | 이완=어깨-귀 거리 큼(baseline이 낮은 분위수면 오히려 "가장 으쓱한 순간"을 baseline으로 잡음) | p90으로 변경, 이유를 docstring에 명시 |

---

## 5. 리포지토리 구조 현황

```
src/dvi/
├─ cli.py                    ✅ 골격 (check-env만 실동작, 나머지는 마일스톤 명시된 스텁)
├─ schemas/                  ✅ 전체 (Frame, PoseSequence, MetricResult, Finding, AnalysisReport ...)
├─ pipeline/                 ✅ Stage 프로토콜, ArtifactStore 캐시, BatchRunner
├─ io/                       ✅ video_reader, video_writer
├─ perception/
│  ├─ pose/                  ✅ base(Pose2D), skeleton(COCO17 매핑), tier_a_vitpose
│  ├─ detect/                ⬜ 빈 패키지 (M4)
│  └─ track/                 ⬜ 빈 패키지 (M4)
├─ geometry/                 ✅ gravity, calibration, com, support_foot, normalize
├─ metrics/                  🟡 posture(6개), axis(4개) — dissociation/couple/rhythm 없음
├─ segment/                  ⬜ 빈 패키지
├─ scoring/                  ⬜ 빈 패키지
├─ report/                   🟡 overlay만 (json_report, html 없음)
└─ reference/                ⬜ 빈 패키지 (M7)

eval/harness.py               🟡 angle_consistency만 (라벨 상관 분석은 라벨 없어서 미착수)
```

---

## 6. 다음으로 자연스러운 작업

우선순위는 대화 중 실사 영상 테스트로 재확인된 순서를 반영한다.

1. **M4: 검출·추적·역할분류** (`perception/detect/`, `perception/track/`) — 실제 커플 영상을 다루려면 사실상 이게 다음 병목. RT-DETRv2(검출) + ByteTrack류(추적) + 클로즈 엠브레이스 대응(SAM2 마스크, ADR-0001 §4 C4-3) 필요. 큰 작업.
2. **ADR-0001 M2 벤치마크 완결** — 사용자가 정면/45°/측면 동시촬영 영상을 준비하고, GVHMR용 SMPL 모델을 직접 등록·다운로드해야 진행 가능.
3. **파이프라인 배선** — 현재 지표 함수들은 전부 독립적으로만 테스트됨. `dvi analyze` CLI가 실제로 Ingest→Pose→Metrics→Report를 엮어 리포트를 뽑도록 연결하는 작업이 아직 없음 (M5 direction).
4. AXI-03/05 — Segment 스테이지(이벤트 검출)를 만들면 자연스럽게 딸려옴.

---

## 7. 검증 방법론 메모

이 프로젝트는 정답 라벨이 없어(PROJECT.md R4) 모든 지표 테스트가 "합성 스켈레톤에 정확한 각도를 넣고 그 값이 나오는지" 확인하는 방식(§7 V1)이다. 이번 세션에서 실제로 이 방식이 버그를 잡아냈다 — 예를 들어 POS-05a의 부호 반전은 합성 테스트가 실패하면서 발견됐고(§4의 버그 5), AXI-04의 무릎각 공식은 직접 손으로 유도한 값과 일치하는지 확인했다. 실제 영상 검증(§4.1)은 "이 숫자가 맞는가"가 아니라 "이 코드가 실사 데이터에서 죽지 않고 그럴듯하게 동작하는가"를 확인하는 별개의 층위였다.
