"""
HumanEval 유틸리티 함수 테스트
"""

import pytest
from utils.humaneval_utils import execute_code, extract_code_from_text


class TestExtractCodeFromText:
    """코드 추출 함수 테스트"""
    
    def test_extract_markdown_code(self):
        """마크다운 형식 코드 추출 테스트"""
        text = """
Here's the solution:

```python
def add_one(x):
    return x + 1
```

This function adds one to the input.
"""
        result = extract_code_from_text(text, "add_one")
        assert "def add_one(x):" in result
        assert "return x + 1" in result
    
    def test_extract_plain_text_code(self):
        """일반 텍스트 형식 코드 추출 테스트"""
        text = """
def multiply(x, y):
    return x * y
"""
        result = extract_code_from_text(text, "multiply")
        assert "def multiply(x, y):" in result
        assert "return x * y" in result
    
    def test_extract_with_entry_point(self):
        """entry_point가 지정된 경우 테스트"""
        text = """
```python
def wrong_function():
    return 0

def correct_function(x):
    return x + 1
```
"""
        result = extract_code_from_text(text, "correct_function")
        assert "def correct_function(x):" in result
        assert "def wrong_function():" not in result
    
    def test_empty_input(self):
        """빈 입력 테스트"""
        assert extract_code_from_text("", "test") == ""
        assert extract_code_from_text(None, "test") == ""


class TestExecuteCode:
    """코드 실행 함수 테스트"""
    
    def test_simple_correct_code(self):
        """간단한 정답 코드 테스트"""
        code = """
def add_one(x):
    return x + 1
"""
        test_cases = """
def check(candidate):
    assert candidate(1) == 2
    assert candidate(0) == 1
    assert candidate(-1) == 0
"""
        score, msg, passed, total = execute_code(code, test_cases, "add_one")
        assert score == 1.0
        assert passed == 1
        assert total == 1
        assert "All tests passed" in msg
    
    def test_incorrect_code(self):
        """잘못된 코드 테스트"""
        code = """
def add_one(x):
    return x  # 잘못된 구현
"""
        test_cases = """
def check(candidate):
    assert candidate(1) == 2
"""
        score, msg, passed, total = execute_code(code, test_cases, "add_one")
        assert score == 0.0
        assert passed == 0
        assert total == 1
    
    def test_syntax_error(self):
        """문법 오류 테스트"""
        code = """
def add_one(x):
    return x +  # 문법 오류
"""
        test_cases = """
def check(candidate):
    assert candidate(1) == 2
"""
        score, msg, passed, total = execute_code(code, test_cases, "add_one")
        assert score == 0.0
        assert "Syntax error" in msg
    
    def test_missing_function(self):
        """함수가 없는 경우 테스트"""
        code = """
def wrong_function(x):
    return x + 1
"""
        test_cases = """
def check(candidate):
    assert candidate(1) == 2
"""
        score, msg, passed, total = execute_code(code, test_cases, "add_one")
        assert score == 0.0
        assert "not found" in msg
    
    def test_invalid_inputs(self):
        """잘못된 입력 테스트"""
        # 빈 코드
        score, msg, passed, total = execute_code("", "def check(c): pass", "test")
        assert score == 0.0
        assert "No code provided" in msg
        
        # 빈 테스트 케이스
        score, msg, passed, total = execute_code("def test(): pass", "", "test")
        assert score == 0.0
        assert "No test code provided" in msg
        
        # 빈 entry_point
        score, msg, passed, total = execute_code("def test(): pass", "def check(c): pass", "")
        assert score == 0.0
        assert "No entry point provided" in msg


if __name__ == "__main__":
    pytest.main([__file__])
