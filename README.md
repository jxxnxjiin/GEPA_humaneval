# HumanEval 프롬프트 최적화

GEPA 프레임워크를 사용하여 HumanEval 코드 생성 문제에 대한 프롬프트를 자동으로 최적화하는 프로젝트입니다.

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd humaneval

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 설정
```

### 2. 실행

```bash
# 기본 실행
python train_humaneval.py

# 커스텀 설정으로 실행
python train_humaneval.py \
    --model_name gpt-4o-mini \
    --reflection_lm gpt-4o \
    --budget 100 \
    --use_merge

# 작은 데이터셋으로 빠른 테스트
python train_humaneval.py \
    --max_dataset_size 50 \
    --budget 20 \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --test_ratio 0.1

# 대용량 데이터셋으로 정확한 평가
python train_humaneval.py \
    --max_dataset_size 1000 \
    --budget 500 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1
```

## 📁 프로젝트 구조

```
humaneval/
├── data.py                    # HumanEval 데이터셋 로딩
├── train_humaneval.py         # 메인 훈련 스크립트
├── utils/
│   └── humaneval_utils.py     # 코드 실행 및 추출 유틸리티
├── prompt-templates/
│   └── instruction_prompt.txt # 시드 프롬프트
├── results/                   # 최적화 결과 저장
└── requirements.txt           # 프로젝트 의존성
```

## ⚙️ 설정 옵션

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--model_name` | gpt-4o-mini | 평가에 사용할 LLM 모델 |
| `--reflection_lm` | gpt-4o | 반성에 사용할 LLM 모델 |
| `--budget` | 100 | 최대 평가 횟수 |
| `--reflection_minibatch_size` | 3 | 반성 배치 크기 |
| `--use_merge` | False | Merge 전략 사용 여부 |
| `--max_merge_invocations` | 5 | 최대 Merge 횟수 |
| `--seed` | 0 | 재현성을 위한 시드 |
| `--max_dataset_size` | None | 최대 데이터셋 크기 (None이면 전체 사용) |
| `--train_ratio` | 0.6 | 훈련 세트 비율 |
| `--val_ratio` | 0.2 | 검증 세트 비율 |
| `--test_ratio` | 0.2 | 테스트 세트 비율 |

## 🎯 주요 기능

- **자동 프롬프트 최적화**: GEPA 프레임워크를 통한 반성 기반 최적화
- **안전한 코드 실행**: HumanEval 형식의 테스트 케이스 완벽 지원
- **상세한 피드백**: 성공/실패 케이스별 맞춤형 피드백 생성
- **Pareto 최적화**: 다중 목표 최적화를 통한 성능 향상
- **Merge 전략**: 우수한 후보들의 결합을 통한 성능 개선

## 📊 결과 해석

최적화 완료 후 `results/` 디렉토리에 다음 정보가 포함된 JSON 파일이 생성됩니다:

- **메타데이터**: 실행 시간, 모델 정보, 데이터셋 정보
- **최적화 결과**: 최고 성능, 최적화된 프롬프트, 평가 히스토리

## 🔧 개발

### 테스트 실행
```bash
python -m pytest tests/
```

### 코드 포맷팅
```bash
black .
flake8 .
```