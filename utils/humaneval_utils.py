import re
import contextlib
import io
from typing import Tuple, List, Optional

# -----------------------------
# 코드 추출
# -----------------------------
def extract_code_from_text(text: str, entry_point: str | None = None) -> str:
    if not text or not isinstance(text, str):
        return ""
    patterns = [
        r"```python\s*(.*?)```",
        r"```\s*(.*?)```",
        r"<code>(.*?)</code>",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            if entry_point and f"def {entry_point}" in match:
                return match.strip()
            if not entry_point:
                return match.strip()

    if entry_point:
        func_match = re.search(
            rf"(def\s+{entry_point}\s*\([^)]*\):[\s\S]+?)(?=\n\S|\Z)", 
            text
        )
        if func_match:
            return func_match.group(1).strip()

    func_match = re.search(r"(def\s+\w+\s*\([^)]*\):[\s\S]+?)(?=\n\S|\Z)", text)
    if func_match:
        return func_match.group(1).strip()
    
    return ""
# -----------------------------
# 코드 실행
# -----------------------------
def execute_code(
    code: str,
    test_code: str,
    entry_point: str,
    extra_imports: Optional[List[str]] = None,
    function_signature: Optional[str] = None
) -> Tuple[float, str, int, int]:
    """
    코드 실행 함수
    Returns: (score, message, passed, total)
    """
    # 입력 검증
    if not code or not isinstance(code, str):
        return 0.0, "No code provided", 0, 0
    if not test_code or not isinstance(test_code, str):
        return 0.0, "No test code provided", 0, 0
    if not entry_point or not isinstance(entry_point, str):
        return 0.0, "No entry point provided", 0, 0

    try:
        # 코드가 이미 추출된 상태인지 확인
        # 만약 코드에 entry_point가 포함되어 있으면 이미 추출된 것으로 간주
        if entry_point in code and code.strip().startswith('def '):
            clean_code = code.strip()
        else:
            # 코드 추출이 필요한 경우
            clean_code = extract_code_from_text(code, entry_point)
            if not clean_code:
                # canonical solution인 경우 함수 시그니처를 추가
                if function_signature and not code.strip().startswith('def '):
                    # 들여쓰기 제거하고 함수 시그니처 추가
                    dedented_code = '\n'.join(line.strip() for line in code.strip().split('\n') if line.strip())
                    clean_code = f"{function_signature}\n    {dedented_code}"
                else:
                    return 0.0, "No code to execute", 0, 0

        # 안전한 builtins 설정
        builtin_names = [
            'len', 'range', 'enumerate', 'zip', 'max', 'min', 'sum', 'abs', 'round',
            'int', 'float', 'str', 'bool', 'list', 'dict', 'set', 'tuple', 'sorted',
            'all', 'any', 'map', 'filter', 'reversed', 'isinstance', 'hasattr',
            'getattr', 'setattr', 'type', 'ord', 'chr', 'pow', 'divmod'
        ]
        
        try:
            if isinstance(__builtins__, dict):
                safe_builtins = {k: __builtins__[k] for k in builtin_names if k in __builtins__}
            else:
                safe_builtins = {k: getattr(__builtins__, k) for k in builtin_names if hasattr(__builtins__, k)}
        except Exception:
            import builtins
            safe_builtins = {k: getattr(builtins, k, None) for k in builtin_names}
            safe_builtins = {k: v for k, v in safe_builtins.items() if v is not None}

        # print 함수 비활성화
        safe_builtins['print'] = lambda *args, **kwargs: None
        namespace = {'__builtins__': safe_builtins}

        # 추가 import 처리
        if extra_imports:
            for imp in extra_imports:
                try:
                    exec(f"import {imp}", namespace)
                except ImportError:
                    pass

        # 출력 캡처
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            # 1. 후보 코드 실행
            try:
                exec(clean_code, namespace)
            except SyntaxError as e:
                return 0.0, f"Syntax error in generated code: {e}", 0, 0
            except Exception as e:
                return 0.0, f"Code execution error: {e}", 0, 0

            # 함수 존재 확인
            if entry_point not in namespace:
                return 0.0, f"Function '{entry_point}' not found in generated code", 0, 0
            if not callable(namespace[entry_point]):
                return 0.0, f"'{entry_point}' is not a callable function", 0, 0

            # 2. 테스트 코드 실행
            try:
                exec(test_code, namespace)
            except Exception as e:
                return 0.0, f"Test code execution error: {e}", 0, 0

            if "check" not in namespace or not callable(namespace["check"]):
                return 0.0, "No check() function found in test cases", 0, 0

            # 3. 테스트 실행
            try:
                namespace["check"](namespace[entry_point])
                passed = 1
            except AssertionError:
                passed = 0
            total = 1

        # 결과 반환
        score = passed / total
        msg = "All tests passed" if passed == total else f"{passed}/{total} test failed"
        return score, msg, passed, total

    except Exception as e:
        return 0.0, f"Runtime error: {e}", 0, 0