import random
import re
from pathlib import Path
from datasets import load_dataset
from typing import TypedDict, Tuple, Dict, List, Optional, Any

# -----------------------------
# Dataset loading
# -----------------------------    
def init_dataset(
    humaneval_dset_name: str = "openai/openai_humaneval",
    max_dataset_size: int | None = None,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    HumanEval 데이터셋을 로드하고 훈련/검증/테스트 세트로 분할합니다.
    
    Args:
        humaneval_dset_name: HumanEval 데이터셋 이름
        max_dataset_size: 최대 데이터셋 크기 (None이면 전체 사용)
        train_ratio: 훈련 세트 비율 (기본값: 0.6)
        val_ratio: 검증 세트 비율 (기본값: 0.2)
        test_ratio: 테스트 세트 비율 (기본값: 0.2)
        
    Returns:
        tuple: (trainset, valset, testset) - 각각 HumanEvalDataInst 리스트
        
    Raises:
        ValueError: 데이터셋 로드 실패 시 또는 비율 합이 1.0이 아닌 경우
        ImportError: datasets 라이브러리가 설치되지 않은 경우
    """

    # 비율 검증
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not (0.99 <= ratio_sum <= 1.01):  # 부동소수점 오차 허용
        raise ValueError(f"비율의 합이 1.0이어야 합니다. 현재: {ratio_sum}")
    
    if not all(0 <= ratio <= 1 for ratio in [train_ratio, val_ratio, test_ratio]):
        raise ValueError("모든 비율은 0과 1 사이여야 합니다.")
    
    train_split = []
    test_split = []

    try:
        # Load the HumanEval dataset
        load_dataset_result = load_dataset(humaneval_dset_name, split="test")
        
        # Convert to list for manipulation
        all_data = list(load_dataset_result)
        
        if not all_data:
            raise ValueError(f"Empty dataset loaded from {humaneval_dset_name}")
        
        # 데이터셋 크기 제한 적용
        if max_dataset_size is not None and max_dataset_size > 0:
            all_data = all_data[:max_dataset_size]
            print(f"📊 데이터셋 크기를 {max_dataset_size}개로 제한했습니다.")
        
        print(f"📊 전체 데이터셋 크기: {len(all_data)}개")
        random.Random(0).shuffle(all_data)
        
    except ImportError as e:
        raise ImportError(
            "datasets library is required. Install with: pip install datasets"
        ) from e
    except Exception as e:
        raise ValueError(f"Failed to load dataset {humaneval_dset_name}: {e}") from e
    
    # 비율 기반 데이터 분할 (최소 1개씩은 보장)
    total_size = len(all_data)
    train_size = max(1, int(total_size * train_ratio))
    val_size = max(1, int(total_size * val_ratio))
    test_size = max(1, total_size - train_size - val_size)  # 나머지는 테스트 세트
    
    # 총 크기가 3개 미만인 경우 조정
    if total_size < 3:
        train_size = 1
        val_size = 1 if total_size >= 2 else 0
        test_size = total_size - train_size - val_size
    
    # 분할 인덱스 계산
    train_end = train_size
    val_end = train_size + val_size
    
    train_data = all_data[:train_end]
    val_data = all_data[train_end:val_end]
    test_data = all_data[val_end:]
    
    print(f"📊 데이터 분할 결과:")
    print(f"   - 훈련 세트: {len(train_data)}개 ({len(train_data)/total_size*100:.1f}%)")
    print(f"   - 검증 세트: {len(val_data)}개 ({len(val_data)/total_size*100:.1f}%)")
    print(f"   - 테스트 세트: {len(test_data)}개 ({len(test_data)/total_size*100:.1f}%)")
    
    # 데이터 변환 함수
    def convert_to_humaneval_format(data_items: list) -> list[dict]:
        """HumanEval 데이터를 표준 형식으로 변환"""
        converted = []
        for item in data_items:
            prompt = item["prompt"]
            canonical_solution = item["canonical_solution"]
            test_cases = item["test"]
            entry_point = item["entry_point"]
            task_id = item["task_id"]
            
            # Extract function signature from prompt
            func_signature_match = re.search(r'def\s+\w+\([^)]*\):', prompt)
            func_signature = func_signature_match.group(0) if func_signature_match else f"def {entry_point}"
            
            converted.append({
                "input": prompt,
                "additional_context": {
                    "canonical_solution": canonical_solution,
                    "test_cases": test_cases,
                    "entry_point": entry_point,
                    "task_id": task_id
                },
                "answer": func_signature
            })
        return converted

    # 각 세트를 표준 형식으로 변환
    trainset = convert_to_humaneval_format(train_data)
    valset = convert_to_humaneval_format(val_data)
    testset = convert_to_humaneval_format(test_data)

    return trainset, valset, testset