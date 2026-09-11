# DanceVideoInstructor — 아키텍처

> 프로젝트 배경·범위·성공 기준은 [PROJECT.md](./PROJECT.md) 참고. 이 문서는 **어떻게 만들 것인가**를 다룬다.

| 항목 | 값 |
| --- | --- |
| 문서 버전 | v0.1 (초안) |
| 상태 | 설계 단계, 코드 미착수 |
| 대상 마일스톤 | M0 ~ M8 (PROJECT.md §6.2) |

---

## 1. 설계 원칙

이후 모든 결정은 이 5개에서 파생된다.

| # | 원칙 | 함의 |
| --- | --- | --- |
| **P1** | **단계 간 계약을 고정한다** | 스테이지는 스키마(§4)로만 소통. 모델 갈아끼워도 하류 코드 불변 |
| **P2** | **모델은 교체 가능한 플러그인** | 포즈 추정은 2년 내 반드시 바뀐다. 어댑터 뒤에 숨긴다 |
| **P3** | **지표는 순수 함수** | `(정규화된 포즈 시계열, 설정) → 지표 시계열`. I/O·모델 의존 없음. 단위 테스트 가능 |
| **P4** | **모든 점수는 역추적 가능** | 점수 → 지표 → 프레임 → 임계값. 블랙박스 채점 금지 |
| **P5** | **인과성을 명시한다** | 각 지표는 causal / non-causal로 태깅. 실시간 확장이 재작성이 아닌 설정 변경이 되도록 |

---

## 2. 시스템 개요

```
┌─────────────────────────────────────────────────────────────────────┐
│  진입점                                                              │
│   CLI (dvi analyze)     FastAPI (로컬)      Webcam Runner (v3)      │
└────────────┬────────────────┬─────────────────────┬─────────────────┘
             │                │                     │
             ▼                ▼                     ▼
        ┌────────────────────────────┐    ┌──────────────────────┐
        │   Batch Pipeline Runner    │    │  Stream Runner (v3)  │
        │   (전체 시계열, 캐시)       │    │  (링버퍼, 인과 처리)  │
        └────────────┬───────────────┘    └──────────┬───────────┘
                     │                                │
                     └──────────┬─────────────────────┘
                                ▼
     ┌──────────────────────────────────────────────────────────┐
     │                    지각 계층 (Perception)                  │
     │  Ingest → Detect → Track → Pose2D → Lift3D → GravityAlign │
     │           → Smooth → Normalize                            │
     └──────────────────────────┬───────────────────────────────┘
                                ▼   PoseSequence (계약, §4.2)
     ┌──────────────────────────────────────────────────────────┐
     │                 분석 계층 (Analysis) — 순수 함수           │
     │  Features → Metrics(POS/AXI/DIS/CPL) → Segment            │
     │           → [Reference DTW] → Rubric Scoring              │
     └──────────────────────────┬───────────────────────────────┘
                                ▼   AnalysisReport (계약, §4.6)
     ┌──────────────────────────────────────────────────────────┐
     │                   표현 계층 (Presentation)                 │
     │  Overlay Video  |  JSON  |  HTML Report  |  Realtime HUD  │
     └──────────────────────────────────────────────────────────┘

     횡단 관심사: Config · Artifact Store(캐시) · Calibration Profile
                 · Logging/Tracing · Eval Harness
```

**핵심 분리**: 지각 계층은 "모델과 GPU의 세계", 분석 계층은 "기하와 규칙의 세계".
둘 사이의 유일한 접점은 `PoseSequence` 한 개다. 이게 지켜지면 지표 개발이 모델과 무관하게 진행된다.

---

## 3. 파이프라인 스테이지

각 스테이지는 `Stage` 인터페이스를 구현하고, 입력/출력 스키마와 인과성 플래그를 선언한다.

| # | 스테이지 | 입력 → 출력 | 핵심 책임 | 인과성 |
| --- | --- | --- | --- | --- |
| 1 | **Ingest** | 파일/스트림 → `Frame` | 디코딩, 회전 메타 보정, 리사이즈, fps 정규화 | causal |
| 2 | **Detect** | `Frame` → `Detection[]` | 인물 바운딩박스 | causal |
| 3 | **Track** | `Detection[]` → `Track[]` | ID 부여·유지, ReID, 가림 복구 | causal (지연 허용) |
| 4 | **Pose2D** | `Frame`+`Track` → `Pose2D` | 2D 키포인트 + 관절별 신뢰도 | causal |
| 5 | **Lift3D** | `Pose2D[윈도우]` → `Pose3D` | 2D→3D 리프팅 (또는 4·5 통합 백엔드) | **non-causal** (윈도우 필요) |
| 6 | **GravityAlign** | `Pose3D` → `Pose3D(world)` | 카메라 좌표 → 중력 정렬 월드 좌표 (§5) | 혼합 |
| 7 | **Smooth** | `Pose3D` → `Pose3D` | 필터링, 결측 보간, 신뢰도 게이팅 | 설정 의존 |
| 8 | **Normalize** | `Pose3D` → `PoseSequence` | 체격 정규화, 골격 정준화, 역할 할당 | causal |
| 9 | **Features** | `PoseSequence` → `FeatureSeries` | 각도·벡터·속도 등 원시 기하량 | causal |
| 10 | **Metrics** | `FeatureSeries` → `MetricResult[]` | 지표 카탈로그 평가 (§6) | 지표별 태그 |
| 11 | **Segment** | `FeatureSeries` → `Segment[]` | 스텝/체중이동 이벤트, 정지/이동 구간 분할 | non-causal |
| 12 | **Reference** | `MetricResult[]`+ref → `Deviation[]` | DTW 정렬 후 편차 (v2) | non-causal |
| 13 | **Scoring** | `MetricResult[]` → `Finding[]`, `Score` | 루브릭 적용, 문제 구간 검출 (§7) | 지표별 태그 |
| 14 | **Report** | 전체 → 산출물 | 오버레이 영상, JSON, HTML | — |

### 3.1 스테이지 인터페이스 (개념)

```python
class Stage(Protocol):
    name: str
    causal: bool                # 실시간 모드에서 사용 가능한가
    lookahead: int              # non-causal일 때 필요한 미래 프레임 수
    input_schema: type
    output_schema: type
    cache_key_fields: list[str] # 캐시 무효화에 쓸 설정 필드

    def run(self, data, ctx: PipelineContext): ...
```

- **배치 러너**: 스테이지를 순차 실행, 각 단계 산출물을 아티팩트 스토어에 캐싱
- **스트림 러너**: `causal=True` 스테이지만 연결, `lookahead > 0`은 링버퍼로 지연 보상 또는 causal 대체 구현으로 스왑

> 이 `causal` 태그가 P5의 구현체다. 실시간 확장 시 "무엇을 못 쓰는지"가 코드에 선언되어 있어야 한다.

---

## 4. 데이터 계약

모든 스키마는 Pydantic 모델로 정의하고 버전 필드를 갖는다. 디스크 직렬화는 Parquet(시계열) + JSON(메타).

### 4.1 좌표계 (가장 중요한 계약)

```python
class Frame(StrEnum):
    IMAGE    = "image"     # 픽셀, y 아래 방향
    CAMERA   = "camera"    # 카메라 중심, 미터, OpenCV 관례
    WORLD_G  = "world_g"   # 중력 정렬: +Y가 중력 반대(위), XZ가 바닥면
    BODY     = "body"      # 골반 원점, 골반선 기준 정렬 (회전 불변 지표용)
```

> **`WORLD_G`가 아니면 "쓰러진 정도"를 말할 수 없다.** 카메라 좌표계 각도는
> 카메라 틸트와 뒤섞여 있어 의미가 없다. 모든 자세/축 지표는 `WORLD_G` 또는 `BODY`를 요구하며,
> 스키마에 프레임을 명시해 잘못된 좌표계 입력 시 런타임 에러를 낸다.

### 4.2 `PoseSequence` — 지각/분석 경계

```python
class PoseSequence:
    schema_version: str
    subject_id: str
    role: Role | None            # LEADER | FOLLOWER | UNKNOWN
    frame: Frame                 # WORLD_G 기대
    fps: float
    t: Float[T]                  # 타임스탬프(초)
    joints: Float[T, J, 3]       # 관절 좌표
    confidence: Float[T, J]      # 0~1
    valid: Bool[T, J]            # 신뢰도 게이팅 + 보간 여부 반영
    skeleton: SkeletonSpec       # 관절 정의(§4.3)
    scale: BodyScale             # 정규화에 쓴 체격 기준값
    provenance: Provenance       # 백엔드명/버전/설정 해시
```

`provenance`는 필수다. 어떤 모델로 뽑은 스켈레톤인지 모르면 지표 비교가 무의미해진다.

### 4.3 `SkeletonSpec` — 골격 정준화

백엔드마다 관절 정의가 다르다(COCO-17, H36M-17, SMPL-24, BlazePose-33).
**내부 정준 골격(DVI-Canonical)** 을 하나 정하고, 각 백엔드는 어댑터에서 여기로 매핑한다.

DVI-Canonical 최소 요구 관절 (자세/축 지표에 필요한 것만):
```
pelvis, spine_mid, neck, head_top, nose(또는 ear_l/ear_r)
shoulder_l/r, elbow_l/r, wrist_l/r
hip_l/r, knee_l/r, ankle_l/r, foot_index_l/r, heel_l/r
```
- 백엔드가 제공하지 않는 관절은 **파생 규칙**으로 채운다 (예: `spine_mid = (pelvis + neck)/2`)
- 파생 관절은 `derived=True`로 표시 → 해당 관절 의존 지표는 신뢰도 하향

### 4.4 `FeatureSeries`

```python
class FeatureSeries:
    subject_id: str
    t: Float[T]
    features: dict[str, Float[T]]   # "torso_pitch_deg", "com_offset_norm", ...
    valid: dict[str, Bool[T]]
```

### 4.5 `MetricResult` / `Finding`

```python
class MetricResult:
    metric_id: str              # "POS-01"
    series: Float[T]
    valid: Bool[T]
    coverage: float             # valid 비율. 낮으면 결론 보류
    target: TargetRange         # 스타일 프로파일에서 주입
    aggregate: dict[str, float] # mean/p50/p95/time_out_of_range 등
    causal: bool

class Finding:
    finding_id: str
    metric_id: str
    span: TimeSpan              # 시작/끝 (초 + 프레임)
    severity: Literal["info","minor","major","critical"]
    evidence: Evidence          # 대표 프레임, 측정값, 임계값, 그래프 크롭
    message_ko: str             # 코칭 코멘트
    confidence: float
```

### 4.6 `AnalysisReport`

```python
class AnalysisReport:
    schema_version: str
    source: SourceInfo          # 파일, 해상도, fps, 길이, 해시
    config_hash: str
    style_profile: str
    subjects: list[SubjectSummary]   # 인물별 점수/지표/발견
    couple: CoupleSummary | None     # CPL-* 지표
    findings: list[Finding]          # 심각도 정렬
    scores: ScoreBreakdown           # 카테고리별 + 종합
    artifacts: dict[str, Path]       # overlay.mp4, metrics.parquet, ...
    warnings: list[str]              # 커버리지 부족, 캘리브레이션 실패 등
```

---

## 5. 지각 계층 상세

### 5.1 포즈 백엔드 전략 (P2)

3개 티어를 어댑터 뒤에 두고 설정으로 선택한다. **어느 것도 지금 확정하지 않는다** (PROJECT.md Q1).

| 티어 | 용도 | 후보 계열 | 특성 |
| --- | --- | --- | --- |
| **A. 경량 2D** | 실시간(v3), 빠른 반복 | MediaPipe Pose, RTMPose(경량) | 빠름 / 3D 없음 → 기울기 지표 신뢰도 낮음 |
| **B. 2D+리프팅** | 오프라인 기본 | 고정확도 2D(RTMPose-x, ViTPose 계열) + 시간축 3D 리프터(MotionBERT 계열) | 정확도/속도 균형, 루트 상대 3D |
| **C. World-grounded** | 축 지표 최고 정밀 | SMPL 기반 비디오 복원(WHAM/GVHMR/TRAM 계열) | **중력 정렬 좌표를 모델이 직접 제공** → §5.2 문제를 상당 부분 해소. 무겁고 라이선스 확인 필요 |

**선정 절차 (M2 이전 수행)**
1. 동일 검증 클립(정면/45°/측면 3각도) 준비
2. 각 백엔드로 상체 기울기 지표 산출
3. 평가 축: 3각도 간 **일관성**(주), 육안 타당성, 처리 속도, 라이선스
4. 결과를 `docs/adr/0001-pose-backend.md`에 기록

> 라이선스 주의: SMPL 모델 파라미터 및 일부 파생 모델은 **연구/비상업 제약**이 있다.
> 백엔드별 라이선스를 `docs/licenses.md`에 매트릭스로 관리한다. (PROJECT.md R7)

어댑터 계약:
```python
class PoseBackend(Protocol):
    name: str
    provides_3d: bool
    provides_world_frame: bool   # True면 GravityAlign 단계 스킵/축소
    skeleton: SkeletonSpec
    def infer(self, frames, tracks) -> RawPose: ...
```

### 5.2 중력 정렬 — 이 프로젝트의 핵심 난제

"상체가 얼마나 쓰러졌나"는 **중력 대비 각도**다. 카메라가 3° 기울어져 찍혔으면 모든 각도가 3° 틀어진다.
`GravityEstimator` 인터페이스에 아래 구현들을 두고, 우선순위 폴백 체인으로 동작시킨다.

| 우선 | 방법 | 설명 | 실패 조건 |
| --- | --- | --- | --- |
| 1 | **모델 제공** | 티어 C 백엔드가 중력 정렬 좌표를 직접 출력 | 티어 C 미사용 |
| 2 | **바닥 평면 피팅** | 시퀀스 전체에서 접지 판정된 발 위치를 모아 평면 RANSAC → 법선 = 중력축 | 접지 샘플 부족(제자리 댄스), 3D 스케일 불안정 |
| 3 | **장축 통계 가정** | 긴 구간의 평균 상체 방향은 대략 수직이라는 사전확률로 회전 추정 | 계속 기울어져 춤추는 스타일 |
| 4 | **수동 캘리브레이션** | UI에서 프레임 위에 수직선(벽/기둥/문틀) 2점 지정, 또는 카메라 틸트 각 직접 입력 | 사용자 개입 필요 |

- 추정 결과는 `CalibrationProfile`로 저장/재사용 (같은 장소·같은 카메라면 재활용)
- 추정 신뢰도를 리포트 `warnings`에 노출. 낮으면 **절대 각도 대신 상대 변화**만 제시

### 5.3 추적과 역할 분류 (커플)

```
Detect → Track(ByteTrack/BoT-SORT 계열) → ReID 임베딩 보강 → RoleAssigner
```

**ID 스와핑**(PROJECT.md R6)이 커플 분석의 최대 적이다. 방어 3중:
1. 궤적 연속성 제약 — 프레임 간 급격한 위치 점프 거부
2. 외형 임베딩 — 가림 이후 재등장 시 매칭
3. **전역 후처리** — 배치 모드에서는 전체 시퀀스를 보고 트랙 조각을 2개 클러스터로 병합 (오프라인 이점 활용)

**역할 분류** (`RoleAssigner`, PROJECT.md Q2) — v1은 휴리스틱 앙상블 + 수동 오버라이드:
- 신장(키) 통계, 초기 진행 방향(리더는 주로 전진 시작), 상대 위치 일관성, 아브라소에서 팔 배치 패턴
- 확신도 낮으면 `UNKNOWN`으로 두고 리포트에서 사용자에게 선택 요청 (틀린 역할로 채점하는 것보다 낫다)

### 5.4 스무딩과 결측 처리

| 모드 | 필터 | 비고 |
| --- | --- | --- |
| 오프라인 | Savitzky-Golay (non-causal) | 위상 왜곡 없음, 미분(속도/가속도) 계산에 유리 |
| 실시간 | One-Euro Filter (causal) | 지연-지터 트레이드오프 조절 가능 |

- 신뢰도 < 임계값인 관절은 `valid=False` → 짧은 결측은 시간축 보간, 긴 결측은 지표에서 제외
- **보간한 값으로 채점하지 않는다.** `valid` 마스크가 지표의 `coverage`로 전파되고,
  coverage가 낮은 구간은 Finding을 생성하지 않거나 `confidence`를 낮춘다

### 5.5 정규화

- **체격 정규화**: 모든 거리 지표를 `torso_length`(pelvis–neck) 또는 신장 추정치로 나눠 무차원화
- **BODY 프레임**: 골반 원점 + 골반선(hip_l→hip_r)으로 요(yaw) 정렬 → 몸이 어느 방향을 보든 동일 지표

---

## 6. 지표 카탈로그

> 이 절이 프로젝트의 실질이다. 각 지표는 독립 모듈 1개 + 단위 테스트 + 설정 블록으로 구성된다.
> **아래 목표값은 전부 가설이다.** PROJECT.md §7 V4/V5로 캘리브레이션 후 갱신한다.

기호: `P_x` = 관절 x의 `WORLD_G` 좌표, `up` = (0,1,0), `mid(a,b)` = 중점

### 6.1 자세 (POS) — "얼마나 쓰러져 있는가"

| ID | 이름 | 계산 | 목표(가설) | 프레임 | 인과 |
| --- | --- | --- | --- | --- | --- |
| **POS-01** | 상체 전후 기울기 | `v = mid(hip_l,hip_r) → neck`; 시상면 투영 후 `up`과의 signed angle (+전방) | 살롱 0~5°, 밀롱게로 5~12°, 누에보 −2~3° | WORLD_G | ✔ |
| **POS-02** | 상체 좌우 기울기 | 동일 `v`를 관상면 투영, `up`과 signed angle | \|θ\| < 3° | WORLD_G | ✔ |
| **POS-03** | 척추 꺾임 (콜랩스) | `∠(hip→spine_mid, spine_mid→neck)`의 180° 편차 | < 8° | WORLD_G | ✔ |
| **POS-04** | 머리 전방 이동 | `(head − neck)`의 전방 성분 / torso_length | < 0.12 | BODY | ✔ |
| **POS-05a** | 어깨 거상 | `1 − dist(shoulder, ear) / baseline` (개인 baseline은 세션 초반 p10) | < 0.10 | BODY | ✔ |
| **POS-05b** | 어깨 비대칭 | 어깨선과 수평면 사이 각 | < 3° | WORLD_G | ✔ |
| **POS-06** | 상체 직선성 | hip→spine_mid→neck→head 4점의 직선 적합 잔차 / torso_length | < 0.05 | WORLD_G | ✔ |

> **POS-01 vs POS-03 구분이 핵심이다.** 탱고에서 "전체가 하나의 판처럼 앞으로 기운 것"(POS-01, 스타일에 따라 정상)과
> "허리에서 꺾여 무너진 것"(POS-03/POS-06, 항상 문제)은 완전히 다른 사건이다.
> 이 둘을 분리 측정하는 게 "상체가 쓰러졌다"를 언어화하는 방법이다.

### 6.2 축·균형 (AXI) — "축을 잡고 있는가"

| ID | 이름 | 계산 | 목표(가설) | 인과 |
| --- | --- | --- | --- | --- |
| **AXI-01** | 무게중심 오프셋 | 추정 COM을 바닥면에 투영 → 지지발 접지점까지 수평 거리 / 신장 | 단일지지 구간 < 0.05 | ✔ |
| **AXI-02** | 축 흔들림 | 정지 구간에서 COM 수평 궤적의 표준편차 및 저크(jerk) RMS | 표준편차 < 0.02 | ✔(윈도우) |
| **AXI-03** | 체중이동 명료성 | 체중이동 이벤트의 전이 시간, 단조성(되돌림 없음), 오버슈트 | 되돌림 0회 | ✖ (구간 필요) |
| **AXI-04** | 지지 무릎 굴곡 | 지지다리 무릎 각 (180=완전신전) | 165~178° (락 금지, 과굴곡 금지) | ✔ |
| **AXI-05** | 접지 품질 | 발 착지 시 발끝→발뒤꿈치 순서 및 접지 시 수직 속도 | 수직 속도 급변 없음 | ✔ |
| **AXI-06** | 축 수직성 | ankle(지지) → COM 벡터와 `up`의 각 | < 4° | ✔ |

- **COM 추정**: 분절 질량비 표준값(Dempster 등)을 관절 위치에 가중 적용. 정밀하진 않지만 **일관적**이면 충분하다(상대 비교용).
- **지지발 판정**: 발목 수직 속도 + 발 높이 + 지속시간의 히스테리시스 상태기계. 이 판정이 AXI 지표 전부의 기반이므로 별도 모듈 + 집중 테스트 대상.

### 6.3 회전·분리 (DIS)

| ID | 이름 | 계산 | 목표(가설) | 인과 |
| --- | --- | --- | --- | --- |
| **DIS-01** | 디소시아시온 | 어깨선 yaw − 골반선 yaw (수평면 투영) | 히로 구간에서 20~45° 확보 | ✔ |
| **DIS-02** | 분리 중 축 유지 | DIS-01이 클 때 POS-02/AXI-01 동시 악화 여부 (상관) | 상관계수 낮을수록 좋음 | ✖ |

### 6.4 커플 (CPL)

| ID | 이름 | 계산 | 목표(가설) | 인과 |
| --- | --- | --- | --- | --- |
| **CPL-01** | 두 축 간 거리 안정성 | 두 사람 골반 간 수평 거리의 시간 분산 | 변동계수 < 0.15 |✔ |
| **CPL-02** | 아브라소 높이 일관성 | 접촉 추정 지점(손/어깨) 높이의 시간 분산 | 낮을수록 좋음 | ✔ |
| **CPL-03** | 상대 상체각 | 두 사람 상체 벡터 사이 각 (V자 개방도) | 스타일 프로파일 의존 | ✔ |
| **CPL-04** | 동기성 | 체중이동 이벤트 시각의 리드-랙 상관 / 평균 시간차 | \|Δt\| < 120ms | ✖ |
| **CPL-05** | 상호 침범 | 한쪽 축 이탈(AXI-01)이 상대 축 이탈을 선행하는지 인과 분석 | 책임 소재 진단용 | ✖ |

> CPL-05가 "본인 문제인가 파트너 문제인가"를 다루는 지표다. 인과 추론은 신중해야 하므로
> v1에서는 **상관과 시간 선행성만 제시**하고 단정하지 않는다.

### 6.5 (선택) 리듬 (RHY) — PROJECT.md Q4

| ID | 이름 | 계산 |
| --- | --- | --- |
| **RHY-01** | 비트 정렬 | 오디오 비트 추정 → 체중이동 이벤트와의 위상차 분포 |

### 6.6 스타일 프로파일

목표 범위는 코드가 아니라 설정에 있다.

```yaml
# configs/styles/salon.yaml
style: salon
targets:
  POS-01: { min: 0.0,  max: 5.0,  unit: deg, tolerance: 2.0 }
  POS-03: { max: 8.0,  unit: deg }
  AXI-01: { max: 0.05, unit: normalized }
  CPL-03: { min: 5.0,  max: 20.0, unit: deg }
weights:
  posture: 0.35
  axis: 0.35
  dissociation: 0.10
  couple: 0.20
```

`configs/styles/`에 `salon.yaml`, `milonguero.yaml`, `nuevo.yaml`. 사용자는 분석 시 프로파일을 선택한다.

---

## 7. 채점과 진단

### 7.1 지표 → 점수

```
1. 각 프레임에서 목표 범위 이탈량 계산 (tolerance 내는 0)
2. 이탈량 → 프레임 페널티 (선형 또는 구간별 매핑, 설정으로 정의)
3. valid 마스크로 가중 평균 → 지표 점수 0~100
4. 카테고리 점수 = 지표 점수의 가중 평균 (스타일 프로파일의 weights)
5. 종합 점수 = 카테고리 점수의 가중 평균
6. coverage < 임계값인 지표는 점수에서 제외하고 warnings에 기록
```

점수는 반드시 **커버리지와 함께** 표시한다. 60% 프레임만 본 점수는 점수가 아니다.

### 7.2 문제 구간 검출 (Finding 생성)

단순 임계값 초과는 노이즈가 많다. 히스테리시스 + 최소 지속시간을 적용한다.

```
enter_threshold > exit_threshold          (채터링 방지)
min_duration                              (순간 스파이크 무시, 예: 0.3초)
merge_gap                                 (가까운 구간 병합, 예: 0.5초)
severity = f(초과량 × 지속시간)
```

각 Finding은 대표 프레임(최대 이탈 지점)과 측정값·임계값을 증거로 첨부한다 → P4.

### 7.3 코칭 코멘트

- v1: **템플릿 기반**. `finding → 메시지 템플릿` 매핑을 `configs/coaching/ko.yaml`에 둔다.
  예: `POS-03 major` → "허리에서 상체가 꺾이고 있습니다. 명치를 들고 골반부터 머리까지 한 선으로…"
- 템플릿에는 측정값이 삽입된다 ("평균 14° 꺾임, 목표 8° 이하").
- LLM 서술 생성은 옵션으로 분리 가능하지만, **수치와 규칙이 원본**이고 LLM은 표현만 담당하도록 한계를 둔다.

### 7.4 레퍼런스 비교 (M7)

```
대상 지표 시계열 ─┐
                  ├─ DTW 정렬 (정렬 특징: 체중이동 이벤트 + 골반 속도) ─→ 구간별 편차 → 히트맵
레퍼런스 시계열 ──┘
```
- 정렬은 **원시 관절 좌표가 아닌 이벤트/지표 수준**에서 수행 (체격·카메라 차이 무효화)
- 레퍼런스 라이브러리: `data/reference/<style>/<clip>/` + 메타(스타일, 템포, 출처, 사용 권한)
- 출력은 "다르다"이지 "틀렸다"가 아니다 — 표현에 주의

---

## 8. 리포지토리 구조

```
DanceVideoInstructor/
├─ README.md
├─ pyproject.toml                  # uv 관리, Python 3.12 고정
├─ docs/
│  ├─ PROJECT.md                   # 프로젝트 정의서
│  ├─ ARCHITECTURE.md              # 이 문서
│  ├─ metrics.md                   # 지표 상세 사양 (§6에서 분리 확장)
│  ├─ licenses.md                  # 모델 라이선스 매트릭스
│  └─ adr/                         # 아키텍처 결정 기록
│     └─ 0001-pose-backend.md
├─ configs/
│  ├─ default.yaml                 # 파이프라인 기본 설정
│  ├─ backends/{tier_a,tier_b,tier_c}.yaml
│  ├─ styles/{salon,milonguero,nuevo}.yaml
│  └─ coaching/ko.yaml
├─ src/dvi/
│  ├─ cli.py                       # typer 진입점
│  ├─ config/                      # 설정 로딩, 스키마, 해시
│  ├─ schemas/                     # §4 데이터 계약 (Pydantic)
│  │  ├─ pose.py  features.py  metrics.py  report.py  frames.py
│  ├─ pipeline/
│  │  ├─ stage.py                  # Stage 프로토콜
│  │  ├─ batch_runner.py
│  │  ├─ stream_runner.py          # v3
│  │  └─ artifacts.py              # 캐시 스토어
│  ├─ io/
│  │  ├─ video_reader.py  video_writer.py  webcam.py
│  ├─ perception/
│  │  ├─ detect/                   # 인물 검출 어댑터
│  │  ├─ track/                    # 추적 + ReID + 전역 후처리
│  │  ├─ pose/
│  │  │  ├─ base.py                # PoseBackend 프로토콜
│  │  │  ├─ tier_a_*.py  tier_b_*.py  tier_c_*.py
│  │  │  └─ skeleton.py            # DVI-Canonical 매핑
│  │  └─ roles.py                  # 리더/팔로워 분류
│  ├─ geometry/
│  │  ├─ frames.py                 # 좌표계 변환
│  │  ├─ gravity.py                # GravityEstimator 폴백 체인
│  │  ├─ calibration.py            # CalibrationProfile
│  │  ├─ smoothing.py              # SavGol / OneEuro
│  │  ├─ com.py                    # 무게중심 추정
│  │  └─ normalize.py
│  ├─ features/                    # 원시 기하량 추출
│  ├─ metrics/
│  │  ├─ registry.py               # 지표 등록/조회
│  │  ├─ posture.py                # POS-*
│  │  ├─ axis.py                   # AXI-*
│  │  ├─ dissociation.py           # DIS-*
│  │  ├─ couple.py                 # CPL-*
│  │  └─ rhythm.py                 # RHY-* (선택)
│  ├─ segment/                     # 스텝/체중이동 이벤트 검출
│  ├─ reference/                   # DTW 비교 (v2)
│  ├─ scoring/                     # 루브릭, Finding 생성, 코칭 템플릿
│  ├─ report/
│  │  ├─ overlay.py                # 스켈레톤+HUD 렌더
│  │  ├─ json_report.py
│  │  └─ html/                     # 정적 리포트 템플릿
│  └─ realtime/                    # v3 전용 어댑터
├─ apps/
│  ├─ api/                         # FastAPI (v2)
│  └─ viewer/                      # 리포트 UI (v2, PROJECT.md Q3)
├─ eval/
│  ├─ harness.py                   # 지표 vs 라벨 상관, 각도 강건성
│  ├─ clips.yaml                   # 검증 클립 목록
│  └─ labels/                      # 전문가 태깅 결과
├─ tests/
│  ├─ unit/                        # 지표 단위 테스트 (합성 스켈레톤)
│  ├─ fixtures/synthetic/          # 기울기·붕괴를 인위 생성한 포즈
│  └─ integration/
├─ scripts/
│  ├─ check_env.py                 # GPU/CUDA/torch 검증 (M0)
│  └─ download_models.py
├─ data/                           # .gitignore
│  ├─ raw/  interim/  reference/  outputs/
└─ models/                         # .gitignore, 가중치
```

---

## 9. 기술 스택과 근거

| 영역 | 선택 | 근거 |
| --- | --- | --- |
| 언어 | **Python 3.12** | ML 생태계. 3.13은 일부 CV/ML 휠 미지원 가능성 → 3.12 고정 |
| 패키지 | **uv** + `pyproject.toml` + 락파일 | 재현성(NFR-03), 설치 속도 |
| 추론 | **PyTorch (CUDA)** / 배포 시 ONNX Runtime·TensorRT | RTX 5070 Ti는 Blackwell 세대 → **CUDA 12.8+ 대응 빌드 필요**. M0의 `check_env.py`에서 실제 호환 조합을 확정하고 락 |
| 영상 I/O | **PyAV** (정확한 타임스탬프) + OpenCV (렌더) | fps/타임스탬프 정확도가 시계열 지표의 전제 |
| 수치 | NumPy, SciPy | SavGol, RANSAC, 신호처리 |
| 스키마 | **Pydantic v2** | 데이터 계약(P1)의 런타임 강제 |
| 시계열 저장 | Parquet (pyarrow) | 지표 재계산·세션 비교 시 효율 |
| CLI | **Typer** | 서브커맨드 구조 |
| API | FastAPI (v2) | 뷰어/실시간 연동 |
| 테스트 | pytest + 합성 픽스처 | P3 지표 순수성 활용 |
| 로깅 | structlog + 단계별 타이밍 | 성능 프로파일링(NFR-01) |

### CLI 초안

```bash
dvi check-env
dvi analyze video.mp4 --style salon --backend tier_b --out out/session01
dvi analyze video.mp4 --resume out/session01        # 캐시된 포즈로 지표만 재계산
dvi calibrate video.mp4 --vertical-hint 2점좌표     # 수동 중력 캘리브레이션
dvi overlay out/session01 --metrics POS-01,AXI-01
dvi compare out/session01 out/session02             # 세션 간 추이
dvi reference-compare out/session01 --ref data/reference/salon/clip03
dvi live --backend tier_a --style salon             # v3
dvi eval run --clips eval/clips.yaml                # 평가 하네스
```

---

## 10. 성능 설계

| 구간 | 전략 |
| --- | --- |
| 디코딩 | 프레임 스트리밍, 분석 해상도로 즉시 다운스케일 |
| 검출/포즈 | 배치 추론, fp16, 필요 시 프레임 서브샘플링 후 보간 |
| 리프팅 | 슬라이딩 윈도우 배치 |
| 지표 | NumPy 벡터화 (프레임 루프 금지) |
| 캐싱 | 스테이지별 아티팩트를 `(입력해시 + 설정해시)` 키로 저장 → 지표 튜닝 반복이 초 단위 |
| 렌더링 | 오버레이는 요청 구간만 렌더 (전체 영상 렌더는 옵션) |

**캐싱이 개발 속도를 결정한다.** 지표 임계값 튜닝은 수백 번 반복되므로,
포즈 추출(느림)과 지표 계산(빠름)의 분리는 성능 최적화가 아니라 **개발 워크플로 요구사항**이다.

---

## 11. 실시간 확장 설계 (v3)

동일 지표 엔진을 재사용하기 위한 제약:

| 항목 | 오프라인 | 실시간 |
| --- | --- | --- |
| 포즈 백엔드 | 티어 B/C | 티어 A |
| 스무딩 | SavGol (non-causal) | One-Euro (causal) |
| 중력 정렬 | 시퀀스 전체 추정 | 세션 시작 시 1회 캘리브레이션 후 고정 |
| 사용 지표 | 전체 | `causal=True`인 것만 (§6 인과 열) |
| 세그먼테이션 | 전체 최적화 | 온라인 상태기계 (지연 허용) |
| 출력 | 리포트 | HUD + 임계값 초과 시 경고 |

`causal=False` 지표(AXI-03, CPL-04 등)는 실시간에서 **자동 비활성화**되고 HUD에 "오프라인 전용"으로 표기된다.
러너만 교체하고 지표 코드는 손대지 않는 것이 목표다.

---

## 12. 테스트 전략

| 레벨 | 대상 | 방법 |
| --- | --- | --- |
| **단위** | 지표 함수 | 합성 스켈레톤(정확히 10° 기울인 몸통)을 입력해 기대 각도 검증 |
| **단위** | 좌표계 변환 | 왕복 변환 항등성, 알려진 회전 적용 검증 |
| **단위** | 지지발 상태기계 | 합성 발목 궤적으로 이벤트 타이밍 검증 |
| **속성(property)** | 지표 불변성 | 포즈를 y축 회전·평행이동·스케일해도 지표 불변 |
| **골든** | 리포트 스키마 | 고정 클립 → JSON 스냅샷 비교 |
| **통합** | 파이프라인 | 짧은 클립 엔드투엔드, 캐시 히트/미스 |
| **평가** | 지표 타당성 | `eval/harness.py` — PROJECT.md §7 V2~V5 |

합성 픽스처가 핵심이다. 실제 영상으로는 "이 값이 맞는지" 판단할 수 없지만,
**내가 만든 10° 기울기를 10°로 읽는지**는 판단할 수 있다.

---

## 13. 구현 순서 (설계 → 코드 진입점)

| 순서 | 작업 | 선행 |
| --- | --- | --- |
| 1 | `scripts/check_env.py` — GPU/CUDA/torch 조합 확정, 의존성 락 | — |
| 2 | `schemas/` 전체 + 좌표계 정의 (§4) | — |
| 3 | `io/video_reader.py` + `pipeline/stage.py` + `batch_runner.py` + 캐시 | 2 |
| 4 | 티어 A 백엔드 1개 + `skeleton.py` 매핑 + 오버레이 렌더 → **M1** | 3 |
| 5 | `tests/fixtures/synthetic/` 합성 스켈레톤 생성기 | 2 |
| 6 | `metrics/posture.py`의 POS-01/03 + 단위 테스트 (합성 입력만으로) | 5 |
| 7 | 티어 B/C 백엔드 벤치마크 → ADR-0001 작성 → **M2** | 4 |
| 8 | `geometry/gravity.py` 폴백 체인 | 7 |
| 9 | 나머지 POS-*, AXI-* + 지지발 상태기계 → **M3** | 6, 8 |
| 10 | 추적·역할 분류·CPL-* → **M4** | 9 |
| 11 | 루브릭·Finding·리포트 → **M5** | 10 |

> **4번과 6번이 병렬 가능**한 것이 이 설계의 이점이다. 합성 스켈레톤으로 지표를 먼저 완성해두면,
> 포즈 백엔드가 확정되는 즉시 실제 영상에 꽂아넣을 수 있다.

---

## 14. 열린 설계 질문

ARCHITECTURE 레벨에서 아직 안 정한 것들 (PROJECT.md §9와 중복 제외):

| # | 질문 | 영향 |
| --- | --- | --- |
| A1 | DVI-Canonical 관절 집합을 H36M-17 기반 vs SMPL 관절 기반 중 무엇으로? → **둘 다 발 관절이 없음.** [ADR-0001 §2 C1](./adr/0001-pose-backend.md)에서 Halpe-26 기반을 제안 | 백엔드 어댑터 난이도, AXI-01/05 성립 여부 |
| A2 | COM 추정에 쓸 분절 질량비 테이블과 3D 관절 부족 시 근사 방식 | AXI-* 전체 정확도 |
| A3 | 지지발 판정 임계값을 고정값 vs 세션 적응형 중 무엇으로? | AXI 지표 안정성 |
| A4 | 캐시 키에 모델 가중치 해시를 포함할지 (포함하면 안전, 무거움) | 재현성 vs 편의 |
| A5 | 리포트 스키마 버저닝/마이그레이션 정책 | 세션 간 비교(FR-16) |
| A6 | 오버레이에 3D 스켈레톤을 재투영할지, 별도 3D 뷰를 둘지 | 표현 계층 복잡도 |
