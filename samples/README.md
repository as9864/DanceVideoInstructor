# 샘플

이 디렉터리는 **실제 GPU/모델/영상 없이** 코드가 무엇을 계산하는지 바로 확인할 수 있는
예제를 모아둔다. `scripts/`의 것들과 달리 여긴 "파이프라인 도구"가 아니라 **읽고 실행해보는
튜토리얼**이 목적이다. `dvi analyze` CLI는 아직 연결되지 않았으므로(`docs/PROGRESS.md` §2,
M3 열 참고) 현재로선 이게 지표 엔진 전체를 한 번에 돌려보는 가장 짧은 경로다.

| 파일 | 필요한 것 | 확인할 수 있는 것 |
| --- | --- | --- |
| [metrics_demo.py](metrics_demo.py) | 없음 (numpy/pydantic/pyyaml만) | POS-\*/AXI-\* 11개 지표 + 각도-일관성(V2) 하네스 |
| [../scripts/tier_a_smoke.py](../scripts/tier_a_smoke.py) | 실제 영상 1개, GPU 권장, 첫 실행 시 ViTPose 가중치(~350MB) 다운로드 | 1인 스켈레톤 추론 + 오버레이 렌더 |
| [../scripts/two_person_axis.py](../scripts/two_person_axis.py) | 커플 영상 1개, GPU 권장 | RT-DETRv2 검출 + 인물별 ViTPose + 2D 기울기 실험 |

---

## `metrics_demo.py`

```bash
uv run python samples/metrics_demo.py
```

### 하는 일

1. `tests/fixtures/synthetic/skeletons.py`의 `build_synthetic_standing_pose()`로
   **정확한 각도를 아는** 3초짜리(90프레임 @30fps) 가상 시퀀스를 만든다 — 이는
   실제 단위 테스트가 지표를 검증하는 방식(`PROJECT.md` §7 V1)과 동일한 기법이다.
   구간 구성:
   - **0~1초**: 정상적인 살롱 자세 (상체 전방 기울기 3°), 좌족 지지
   - **1~2초**: 의도적인 "붕괴" — 상체 기울기(POS-01)와 척추 꺾임(POS-03)을
     **독립적으로** 동시에 키우고, 지지 무릎을 가라앉힌다 (우족 지지)
   - **2~3초**: 회복, 좌족 지지로 복귀
2. `src/dvi/metrics/posture.py`·`axis.py`의 실제 함수로 POS-01/02/03/04/05a/05b/06,
   AXI-01/02/04/06 11개 지표를 계산하고, `configs/styles/salon.yaml`의 목표 범위와
   비교해 **구간을 벗어난 프레임 비율**을 함께 찍는다.
3. `eval/harness.py`의 `angle_consistency()`(ADR-0001 §5 / PROJECT.md §7 V2 방법론)를
   합성 노이즈로 시연한다 — "정면/45°/측면 세 각도에서 같은 동작을 찍으면 지표가
   얼마나 흔들리는가"라는 실제 질문의 **하네스 사용법**만 보여준다 (진짜 카메라
   강건성 결과가 아님. 그건 동시 다각도 촬영 영상이 있어야 나온다).

### 실행 결과 (예시, `numpy.random.default_rng(0)`로 고정된 노이즈 부분 제외 나머지는 결정적)

```
synthetic 3-second sequence: normal stance -> deliberate torso collapse -> recovery
90 frames @ 30 fps, style profile = salon

metric   unit         coverage     mean      p95  out-of-range
--------------------------------------------------------------
POS-01   deg            100.0%    5.167    8.624         48.9%
POS-02   deg            100.0%    0.000    1.989          0.0%
POS-03   deg            100.0%    5.000   13.110         31.1%
POS-04   normalized     100.0%    0.045    0.075          0.0%
POS-05a  normalized     100.0%    0.001    0.002          0.0%
POS-05b  deg            100.0%    0.000    0.000          0.0%
POS-06   normalized     100.0%    0.009    0.024          0.0%
AXI-01   normalized     100.0%    0.089    0.093        100.0%
AXI-02   normalized     100.0%    0.933    3.360        100.0%
AXI-04   deg            100.0%  177.333  180.000         51.1%
AXI-06   deg            100.0%    8.800    9.266        100.0%

angle_consistency(POS-01, 3 synthetic viewpoints: ['front', '45deg', 'side']):
  mean_std=1.75 deg, max_std=5.46 deg, coverage=100.0%
  (this only demonstrates the harness on synthetic noise, not a real camera-robustness result — ADR-0001 §5 needs real simultaneous multi-angle footage to answer that.)
```

### 이 출력을 읽는 법 (그리고 함정)

- **POS-01/POS-03이 같이 out-of-range로 뜨는 것이 의도된 결과다.** §6.1의 핵심 주장
  (전신이 기운 것과 허리가 꺾인 것은 다른 사건)을 시연하려고 두 각도를 독립적으로
  키웠다 — 실제 영상에서 둘 중 하나만 움직인다면 "무너짐"이 아니라 "스타일상
  정상적인 기울기"거나 "국소적인 꺾임만"일 수 있다는 뜻이다.
- **AXI-01/AXI-06이 "정상 구간"에서도 계속 100% out-of-range다 — 이건 버그가 아니라
  합성 픽스처의 한계다.** `build_synthetic_standing_pose()`는 골반을 항상 몸 중심에
  두고 다리만 양옆으로 벌리므로, 한쪽 발로만 지지해도 무게중심이 실제로 그 발
  위로 옮겨가지 않는다 (진짜 사람은 한 발로 설 때 골반 자체를 지지발 쪽으로
  이동시킨다). 그래서 이 픽스처로는 "좋은 축"이 살롱 프로파일의 0.05/4° 문턱을
  절대 만족하지 못한다 — 지표 계산 자체가 틀렸다는 뜻이 아니라, **이 데모용
  픽스처가 "축이 좋은 자세"를 표현할 만큼 정교하지 않다는 뜻**이다. 진짜 기하학적
  영점 검증은 `tests/unit/test_metrics_axis.py`가 무게중심을 지지발 바로 위로
  옮긴 경우를 별도로 만들어서 한다.
- 목표 범위(`configs/styles/*.yaml`) 자체가 전부 **가설**이다 (`PROJECT.md` §7 V4/V5로
  전문가 라벨 기반 캘리브레이션이 되기 전까지는 신뢰하면 안 된다). out-of-range
  비율은 "이 시퀀스가 나쁘다"가 아니라 "현재 가설 문턱과 비교하면 이렇다"로 읽어야
  한다.

### 코드를 바꿔서 실험해보기

- `_segment_params()`의 각도 값을 바꾸면 어떤 지표가 어떻게 반응하는지 바로 확인할
  수 있다 — 예를 들어 `roll`(POS-02)만 키우고 `pitch`/`kyphosis`는 그대로 두면
  POS-01/POS-03은 안 움직이고 POS-02/POS-05b만 움직이는지 보라.
- `configs/styles/milonguero.yaml`이나 `nuevo.yaml`을 `load_style_targets()`에
  넘겨서 같은 시퀀스가 스타일별로 다르게 채점되는지 비교해보라 (밀롱게로는
  전방 기울기 허용치가 훨씬 넓다).

---

## `scripts/tier_a_smoke.py`, `scripts/two_person_axis.py`

둘 다 진짜 영상과 GPU가 필요한 "수동 스모크 테스트"다 (`scripts/*.py`의 각 docstring
참고). 자동 테스트 스위트(`pytest`)에는 포함되지 않는다 — 실행 예시:

```bash
uv run python scripts/tier_a_smoke.py data/raw/sample1_front.mp4
uv run python scripts/two_person_axis.py data/raw/sample1_front.mp4 --roi 100,50,800,600
```

`data/raw/*.mp4`는 이 저장소에 실제로 포함된 샘플 영상이다 (1인 정면 촬영,
`docs/PROGRESS.md` §4.1 참고). `two_person_axis.py`는 아직 프레임 간 ID 추적이
없으므로(M4 이전) 커플 영상에서 인물 순서가 프레임마다 바뀔 수 있다 — 스크립트
자체 docstring이 이를 명시한다.
