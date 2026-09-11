# DanceVideoInstructor (dvi)

아르헨티나 탱고 영상에서 포즈 스켈레톤을 추출해 자세(상체 정렬)와 축(axis/balance)을
정량화하고, 문제 구간을 근거 프레임과 함께 제시하는 분석 도구.

- 프로젝트 정의: [docs/PROJECT.md](docs/PROJECT.md)
- 아키텍처: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- 포즈 백엔드 결정: [docs/adr/0001-pose-backend.md](docs/adr/0001-pose-backend.md)
- 모델 라이선스: [docs/licenses.md](docs/licenses.md)

## 상태

설계 완료, M0(스캐폴드) 진행 중. 코드는 아직 어떤 지표도 실제로 계산하지 않는다.

## 개발 환경

```bash
uv sync
uv run dvi check-env
uv run dvi --help
```

Python 3.12 고정 (`uv python pin 3.12`), GPU는 RTX 50xx(Blackwell) 기준 CUDA 12.8+ 빌드 필요.
