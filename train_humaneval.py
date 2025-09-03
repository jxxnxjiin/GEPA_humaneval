import sys
import os
import json
from datetime import datetime
import argparse
from pathlib import Path
from pydantic import BaseModel, Field
import litellm
from gepa import optimize
from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from gepa.logging.logger import StdOutLogger
from typing import TypedDict, Any, Protocol
from utils.humaneval_utils import extract_code_from_text, execute_code
from data import init_dataset
import contextlib
# -----------------------------
# HumanEval adapter
# -----------------------------
class HumanEvalDataInst(TypedDict):
    input: str
    additional_context: dict[str, str]
    answer: str

class HumanEvalTrajectory(TypedDict):
    data: HumanEvalDataInst
    full_assistant_response: str

class HumanEvalRolloutOutput(TypedDict):
    full_assistant_response: str

class HumanEvalStructuredOutput(BaseModel):
    code_solution: str = Field(
        ..., description="The complete Python function implementation that solves the given problem"
    )
    explanation: str = Field(
        ..., description="A brief explanation of the approach used to solve the problem"
    )

class HumanEvalAdapter(GEPAAdapter[HumanEvalDataInst, HumanEvalTrajectory, HumanEvalRolloutOutput]):
    """
    HumanEval 데이터셋을 위한 GEPA 어댑터 구현.
    
    GEPA 프레임워크를 사용하여 HumanEval 코드 생성 문제에 대한 프롬프트 최적화를 수행합니다.
    """
    
    def __init__(self, model: str, failure_score: float = 0.0, api_key: str | None = None):
        """
        Args:
            model: 평가에 사용할 LLM 모델명
            failure_score: 실패한 케이스에 부여할 점수 (기본값: 0.0)
            api_key: LLM API 키
        """
        self.model = model
        self.failure_score = failure_score
        self.api_key = api_key

    def _call_llm(self, messages: list[dict], timeout: int = 30) -> str:
        """LLM 호출을 위한 공통 함수"""
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
                timeout=timeout
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: Failed to get response from model - {str(e)}"

    def evaluate(
        self,
        batch: list[HumanEvalDataInst],
        candidate: dict[str, str],
        capture_traces: bool = True,
    ) -> EvaluationBatch[HumanEvalTrajectory, HumanEvalRolloutOutput]:
        """
        HumanEval 배치에 대해 후보 프로그램을 평가합니다.
        
        Args:
            batch: 평가할 HumanEval 데이터 인스턴스 목록
            candidate: 평가할 후보 프로그램 (컴포넌트명 -> 컴포넌트 텍스트)
            capture_traces: True인 경우 상세한 추적 정보를 캡처
            
        Returns:
            평가 결과를 포함한 EvaluationBatch
            
        Note:
            - 개별 예제 실패 시 예외를 발생시키지 않고 failure_score를 반환
            - 시스템적 실패(모델 없음, 설정 오류 등)에만 예외 발생
        """

        outputs, scores, trajectories = [], [], [] if capture_traces else None

        if not candidate:
            raise ValueError("Candidate must contain at least one component text.")

        system_content = candidate.get("instruction_prompt", "")
        for data in batch:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": data['input']}
            ]

            raw_text = self._call_llm(messages)

            entry_point = data["additional_context"].get("entry_point", "")
            code_solution = extract_code_from_text(raw_text, entry_point)

            output_text = f"Assistant output:\n{raw_text}\n\nExtracted code:\n{code_solution}"

            test_cases = data["additional_context"].get("test_cases", "")

            # 디버깅 정보 추가
            debug_info = f"\nDebug Info:\n- Entry point: {entry_point}\n- Code extracted: {len(code_solution) > 0}\n- Test cases available: {len(test_cases) > 0}"

            if code_solution and test_cases and entry_point:
                # 함수 시그니처 추출 (prompt에서)
                import re
                prompt = data['input']
                func_signature_match = re.search(r'def\s+\w+\([^)]*\)\s*->\s*[^:]+:', prompt)
                if not func_signature_match:
                    func_signature_match = re.search(r'def\s+\w+\([^)]*\):', prompt)
                function_signature = func_signature_match.group(0) if func_signature_match else None
                
                score, err_msg, _, _ = execute_code(code_solution, test_cases, entry_point, function_signature=function_signature)
                debug_info += f"\n- Execution score: {score}\n- Error message: {err_msg}"
                if score < 1.0:
                    output_text += f"\nExecution Result: {err_msg}"
            else:
                score = self.failure_score
                if not code_solution:
                    output_text += "\nError: No code could be extracted from response"
                elif not test_cases:
                    output_text += "\nError: No test cases available"
                elif not entry_point:
                    output_text += "\nError: No entry point specified"
            
            output_text += debug_info

            outputs.append({"full_assistant_response": output_text})
            scores.append(score)

            if capture_traces:
                trajectories.append({"data": data, "full_assistant_response": output_text})

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[HumanEvalTrajectory, HumanEvalRolloutOutput],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """
        반성 데이터셋을 구축하여 instruction refinement를 위한 데이터를 생성합니다.
        
        Args:
            candidate: evaluate()에서 평가된 동일한 후보 프로그램
            eval_batch: evaluate(..., capture_traces=True)의 결과
            components_to_update: 업데이트할 컴포넌트 목록
            
        Returns:
            컴포넌트명 -> 반성 데이터셋 레코드 목록 매핑
            각 레코드는 JSON 직렬화 가능하며 다음 스키마를 따릅니다:
            {
                "Inputs": str,           # 컴포넌트 입력의 최소한의 깔끔한 뷰
                "Generated Outputs": str, # 모델 출력 또는 원시 텍스트
                "Feedback": str          # 성능에 대한 피드백 (정답, 오류 메시지 등)
            }
            
        Note:
            - 반성 데이터셋은 teacher LLM이 개선된 컴포넌트 텍스트를 제안하는 데 사용됩니다
            - 성공/실패 케이스 모두에 대해 구체적이고 학습 중심적인 피드백을 제공합니다
        """
        ret_d: dict[str, list[dict[str, Any]]] = {}

        assert len(components_to_update) == 1
        comp = components_to_update[0]

        items: list[dict[str, Any]] = []
        trace_instances = list(zip(eval_batch.trajectories, eval_batch.scores, eval_batch.outputs, strict=False))

        for trace_instance in trace_instances:
            traj, score, _ = trace_instance
            data = traj["data"]
            generated_outputs = traj["full_assistant_response"]

            if score > 0.0:
                # 성공 케이스: 더 구체적인 피드백 제공
                feedback = self._generate_success_feedback(data, generated_outputs)
            else:
                # 실패 케이스: 상세한 오류 분석과 개선 방향 제시
                feedback = self._generate_failure_feedback(data, generated_outputs)

            d = {"Inputs": data["input"], "Generated Outputs": generated_outputs, "Feedback": feedback}
            items.append(d)

        ret_d[comp] = items

        if len(items) == 0:
            raise Exception("No valid predictions found for any module.")

        return ret_d

    def _generate_success_feedback(self, data: dict, generated_outputs: str) -> str:
        """성공 케이스에 대한 구체적인 피드백 생성"""
        entry_point = data["additional_context"].get("entry_point", "")
        
        feedback_parts = [
            "✅ SUCCESS: The generated code solution is correct and passes all test cases.",
            f"✅ Function '{entry_point}' was implemented correctly.",
            "✅ All assertions in the test cases passed.",
            "",
            "💡 Key strengths of this solution:",
            "- Correct function signature and naming",
            "- Proper logic implementation", 
            "- Handles all edge cases correctly",
            "",
            "🎯 This is a good example of how to approach similar problems."
        ]
        
        return "\n".join(feedback_parts)

    def _generate_failure_feedback(self, data: dict, generated_outputs: str) -> str:
        """실패 케이스에 대한 상세한 피드백 생성"""
        test_cases = data["additional_context"].get("test_cases", "")
        entry_point = data["additional_context"].get("entry_point", "")
        canonical_solution = data["additional_context"].get("canonical_solution", "")
        
        # 코드 추출 시도
        from utils.humaneval_utils import extract_code_from_text
        extracted_code = extract_code_from_text(generated_outputs, entry_point)
        
        feedback_parts = [
            "❌ FAILURE: The generated code solution is incorrect or fails test cases.",
            "",
            "📋 Problem Analysis:",
            f"- Function name: {entry_point}",
            f"- Generated code length: {len(extracted_code)} characters" if extracted_code else "- No valid code could be extracted",
            "",
            "🔍 Common issues to check:",
            "1. Function signature matches the expected name and parameters",
            "2. Logic correctly implements the problem requirements", 
            "3. Edge cases are handled properly",
            "4. Return type matches expectations",
            "5. No syntax errors or runtime exceptions",
            ""
        ]
        
        if test_cases:
            feedback_parts.extend([
                "🧪 Test Cases to Pass:",
                f"```python\n{test_cases}\n```",
                ""
            ])
        
        if extracted_code:
            feedback_parts.extend([
                "📝 Your Generated Code:",
                f"```python\n{extracted_code}\n```",
                ""
            ])
        
        if canonical_solution:
            feedback_parts.extend([
                "✅ Reference Solution:",
                f"```python\n{canonical_solution}\n```",
                "",
                "💡 Study the reference solution to understand:",
                "- The correct approach to solve this problem",
                "- How edge cases are handled",
                "- The expected function signature and behavior"
            ])
        
        return "\n".join(feedback_parts)

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """
        GEPA의 기본 instruction proposal 로직을 사용합니다.
        
        사용자 정의 proposal 로직이 필요한 경우 이 메서드를 오버라이드할 수 있습니다.
        예: 다른 LLM 사용, DSPy 시그니처 구현, 여러 컴포넌트 동시 업데이트 등
        
        Args:
            candidate: 현재 후보 프로그램 (컴포넌트명 -> 컴포넌트 텍스트)
            reflective_dataset: make_reflective_dataset에서 생성된 반성 데이터셋
            components_to_update: 업데이트할 컴포넌트 목록
            
        Returns:
            컴포넌트명 -> 새로운 컴포넌트 텍스트 매핑
        """
        # GEPA의 기본 구현을 사용 (None 반환 시 기본 로직 사용)
        return {}

# -----------------------------
# Main
# -----------------------------

def main():
    """메인 실행 함수"""
    # 타임스탬프 생성 (결과 파일과 동일하게)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # results 디렉토리 생성
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    
    # 로그 파일 경로
    log_file = results_dir / f"training_log_{timestamp}.log"
    
    print("=" * 60)
    print("🚀 HumanEval 프롬프트 최적화 시작")
    print("=" * 60)
    print(f"📝 로그 파일: {log_file}")
    
    # 로그 파일에 시작 메시지 기록
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {'=' * 60}\n")
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 🚀 HumanEval 프롬프트 최적화 시작\n")
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {'=' * 60}\n")
    
    # 로그 파일에 모든 출력을 기록하는 함수
    def log_print(*args, **kwargs):
        original_print(*args, **kwargs)
        with open(log_file, 'a', encoding='utf-8') as f:
            original_print(*args, file=f, **kwargs)
    
    # 기존 print 함수를 log_print로 교체
    import builtins
    original_print = builtins.print
    builtins.print = log_print
    
    # Step 1: 환경 설정 및 인자 파싱
    print("\n📋 Step 1: 환경 설정 및 인자 파싱")
    print("-" * 40)
    
    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ .env 파일에서 환경 변수 로드 완료")
        
    except ImportError:
        print("⚠️  Warning: python-dotenv not installed. Please install with 'pip install python-dotenv' or set environment variables manually.")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY not set in environment")
    print("✅ OpenAI API 키 확인 완료")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="Model name (e.g., gpt-3.5-turbo, gpt-4, claude-3-sonnet-20240229)")
    parser.add_argument("--humaneval_dset_name", type=str, default="openai/openai_humaneval")
    parser.add_argument("--budget", type=int, default=100, help="The budget for the optimization process.")
    parser.add_argument(
        "--reflection_lm", type=str, default="gpt-4o", help="The name of the reflection LM to use."
    )
    parser.add_argument(
        "--reflection_minibatch_size", type=int, default=5, help="The size of the minibatch for the reflection LM."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="The seed for the random number generator for reproducibility."
    )
    parser.add_argument(
        "--use_merge", action="store_true", help="Whether to use the merge strategy for candidate combination."
    )
    parser.add_argument(
        "--max_merge_invocations", type=int, default=5, help="Maximum number of merge invocations to perform."
    )
    parser.add_argument(
        "--max_dataset_size", type=int, default=None, help="Maximum dataset size to use (None for full dataset)."
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.5, help="Training set ratio (default: 0.5)."
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.3, help="Validation set ratio (default: 0.3)."
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.2, help="Test set ratio (default: 0.2)."
    )
    args = parser.parse_args()
    
    print(f"📊 설정된 파라미터:")
    print(f"   - 모델: {args.model_name}")
    print(f"   - 데이터셋: {args.humaneval_dset_name}")
    print(f"   - 예산: {args.budget}")
    print(f"   - 반성 모델: {args.reflection_lm}")
    print(f"   - 반성 배치 크기: {args.reflection_minibatch_size}")
    print(f"   - 시드: {args.seed}")
    print(f"   - Merge 전략: {args.use_merge}")
    if args.use_merge:
        print(f"   - 최대 Merge 횟수: {args.max_merge_invocations}")
    print(f"   - 최대 데이터셋 크기: {args.max_dataset_size or '전체'}")
    print(f"   - 데이터 분할 비율: 훈련 {args.train_ratio:.1%}, 검증 {args.val_ratio:.1%}, 테스트 {args.test_ratio:.1%}")
    
    # Step 2: 프롬프트 템플릿 및 데이터셋 로딩
    print("\n📚 Step 2: 프롬프트 템플릿 및 데이터셋 로딩")
    print("-" * 40)
    
    INSTRUCTION_PROMPT_PATH = Path(__file__).parent / "prompt-templates/instruction_prompt.txt"
    seed_instruction = INSTRUCTION_PROMPT_PATH.read_text()
    print(f"✅ 시드 프롬프트 로드 완료: {INSTRUCTION_PROMPT_PATH}")
    print(f"   프롬프트 길이: {len(seed_instruction)} 문자")
    
    # 시드 프롬프트를 로그에 기록
    print("\n📝 시드 프롬프트:")
    print("-" * 40)
    print(seed_instruction)
    print("-" * 40)

    print(f"📥 데이터셋 로딩 중: {args.humaneval_dset_name}")
    trainset, valset, testset = init_dataset(
        humaneval_dset_name=args.humaneval_dset_name,
        max_dataset_size=args.max_dataset_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    print("✅ 데이터셋 로딩 완료")

    # Step 3: 어댑터 및 반성 모델 설정
    print("\n🔧 Step 3: 어댑터 및 반성 모델 설정")
    print("-" * 40)
    
    reflection_lm_name = args.reflection_lm
    adapter_model = args.model_name
    budget = args.budget
    reflection_minibatch_size = args.reflection_minibatch_size
    seed = args.seed

    def reflection_lm(prompt: str):
        """GEPA API 방식의 reflection language model 호출"""
        try:
            response = litellm.completion(
                model=reflection_lm_name,
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
                timeout=30
            )
            return response.choices[0].message.content
        except Exception as e:
            # GEPA 엔진이 에러를 처리하도록 예외를 다시 발생시킴
            raise RuntimeError(f"Reflection LM error: {e}") from e

    # GEPA 표준 로거 설정
    logger = StdOutLogger()
    
    print(f"✅ 어댑터 설정 완료:")
    print(f"   - 평가 모델: {adapter_model}")
    print(f"   - 반성 모델: {reflection_lm_name}")
    print(f"   - 반성 배치 크기: {reflection_minibatch_size}")
    print(f"   - 시드: {seed}")
    print(f"   - 로거: GEPA StdOutLogger")

    # Step 4: 최적화 실행
    print("\n🚀 Step 4: 프롬프트 최적화 실행")
    print("-" * 40)
    print(f"⏱️  예상 소요 시간: {budget}회 평가 (약 {budget // 2}분)")
    print("🔄 최적화 시작...")
    
    start_time = datetime.now()
    
    try:
        optimized_results = optimize(
            seed_candidate={"instruction_prompt": seed_instruction},
            trainset=trainset,
            valset=valset,
            adapter=HumanEvalAdapter(
                model=adapter_model, 
                api_key=api_key
            ),
            reflection_lm=reflection_lm,
            reflection_minibatch_size=reflection_minibatch_size,
            perfect_score=0.95,  # HumanEval에서 현실적인 목표
            skip_perfect_score=False,  # Perfect score 달성해도 계속 최적화
            candidate_selection_strategy="pareto",  # GEPA API 권장: Pareto frontier 활용
            use_wandb=False,
            max_metric_calls=budget,
            seed=seed,
            display_progress_bar=True,
            raise_on_exception=True,  # GEPA API 권장: 예외 발생 시 중단
            logger=logger,  # GEPA 표준 로거 사용
            use_merge=args.use_merge,  # GEPA API: Merge 전략 사용 여부
            max_merge_invocations=args.max_merge_invocations  # GEPA API: 최대 Merge 횟수
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"✅ 최적화 완료!")
        print(f"⏱️  총 소요 시간: {duration}")
        
        # GEPAResult 객체의 속성 확인 및 출력
        print(f"📊 결과 객체 타입: {type(optimized_results)}")
        print(f"📊 결과 객체 속성: {[attr for attr in dir(optimized_results) if not attr.startswith('_')]}")
        
        # GEPAResult 객체에서 최종 점수 추출
        if hasattr(optimized_results, 'val_aggregate_scores') and optimized_results.val_aggregate_scores:
            best_score = max(optimized_results.val_aggregate_scores)
            print(f"📊 최종 점수: {best_score:.4f}")
        elif hasattr(optimized_results, 'best_score'):
            print(f"📊 최종 점수: {optimized_results.best_score:.4f}")
        else:
            print(f"📊 최종 점수: 정보 없음")
        
        # 최적화된 프롬프트 추출 및 출력
        optimized_prompt = None
        if hasattr(optimized_results, 'candidates') and optimized_results.candidates:
            # 최고 점수를 가진 후보 찾기
            best_idx = getattr(optimized_results, 'best_idx', 0)
            if best_idx < len(optimized_results.candidates):
                best_candidate = optimized_results.candidates[best_idx]
                optimized_prompt = best_candidate.get('instruction_prompt', '')
        
        if optimized_prompt:
            print("\n📝 최적화된 프롬프트:")
            print("-" * 40)
            print(optimized_prompt)
            print("-" * 40)
            
            # 프롬프트 변화 분석
            print("\n🔄 프롬프트 변화 분석:")
            print("-" * 40)
            if optimized_prompt != seed_instruction:
                print("✅ 프롬프트가 최적화되었습니다!")
                print(f"   - 시드 프롬프트 길이: {len(seed_instruction)} 문자")
                print(f"   - 최적화된 프롬프트 길이: {len(optimized_prompt)} 문자")
                print(f"   - 길이 변화: {len(optimized_prompt) - len(seed_instruction):+d} 문자")
                
                # 주요 변화점 분석
                if len(optimized_prompt) > len(seed_instruction):
                    print("   - 프롬프트가 더 상세해졌습니다")
                elif len(optimized_prompt) < len(seed_instruction):
                    print("   - 프롬프트가 더 간결해졌습니다")
                else:
                    print("   - 프롬프트 길이는 동일하지만 내용이 변경되었습니다")
            else:
                print("ℹ️  프롬프트가 변경되지 않았습니다 (시드 프롬프트가 최적 상태)")
        else:
            print("\n⚠️  최적화된 프롬프트를 찾을 수 없습니다")

    except Exception as e:
        print(f"❌ 최적화 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 5: 결과 저장
    print("\n💾 Step 5: 결과 저장")
    print("-" * 40)
    
    # results 디렉토리 생성
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"optimized_results_{timestamp}.json"
    
    # 상세한 결과 정보를 포함한 저장
    result_data = {
        "metadata": {
            "timestamp": timestamp,
            "duration_seconds": duration.total_seconds(),
            "model_name": adapter_model,
            "reflection_lm": reflection_lm_name,
            "budget": budget,
            "seed": seed,
            "dataset_info": {
                "name": args.humaneval_dset_name,
                "train_size": len(trainset),
                "val_size": len(valset),
                "test_size": len(testset)
            }
        },
        "optimization_results": optimized_results.to_dict()
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 결과 저장 완료: {output_file}")
    print(f"📁 파일 크기: {output_file.stat().st_size / 1024:.1f} KB")
    
    # 최종 요약
    print("\n" + "=" * 60)
    print("🎉 HumanEval 프롬프트 최적화 완료!")
    print("=" * 60)
    # 최종 성능 출력 (GEPAResult 객체에서 추출)
    if hasattr(optimized_results, 'val_aggregate_scores') and optimized_results.val_aggregate_scores:
        best_score = max(optimized_results.val_aggregate_scores)
        print(f"📊 최종 성능: {best_score:.4f}")
    elif hasattr(optimized_results, 'best_score'):
        print(f"📊 최종 성능: {optimized_results.best_score:.4f}")
    else:
        print(f"📊 최종 성능: 정보 없음")
    
    # 최적화된 프롬프트를 최종 요약에 포함
    if optimized_prompt:
        print("\n📝 최적화된 프롬프트:")
        print("-" * 40)
        print(optimized_prompt)
        print("-" * 40)
    
    print(f"⏱️  총 소요 시간: {duration}")
    print(f"💾 결과 파일: {output_file}")
    print(f"📝 로그 파일: {log_file}")
    print("=" * 60)
    
    # 로그 파일에 완료 메시지 및 최적화된 프롬프트 기록
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ⏱️  총 소요 시간: {duration}\n")
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 💾 결과 파일: {output_file}\n")
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 📝 로그 파일: {log_file}\n")
        
        # 최적화된 프롬프트를 로그에 기록
        if optimized_prompt:
            f.write(f"\n📝 최적화된 프롬프트:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{optimized_prompt}\n")
            f.write("-" * 40 + "\n")
        
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {'=' * 60}\n")
    
    # print 함수 복원
    builtins.print = original_print
    
    print(f"\n✅ 로그가 저장되었습니다: {log_file}")


if __name__ == "__main__":
    main()