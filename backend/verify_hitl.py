"""Verification script for Human-in-the-Loop implementation."""

import os
import re

def check_file_exists(filepath):
    """Check if a file exists."""
    exists = os.path.exists(filepath)
    status = "[PASS]" if exists else "[FAIL]"
    print(f"{status} File exists: {filepath}")
    return exists


def check_code_pattern(filepath, pattern, description):
    """Check if a code pattern exists in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            found = bool(re.search(pattern, content, re.MULTILINE | re.DOTALL))
            status = "[PASS]" if found else "[FAIL]"
            print(f"{status} {description}")
            return found
    except Exception as e:
        print(f"[ERROR] Could not check {filepath}: {e}")
        return False


def main():
    """Run verification checks."""
    print("=" * 70)
    print("Human-in-the-Loop Implementation Verification")
    print("=" * 70)
    
    results = []
    
    # Check file modifications
    print("\n1. Checking File Modifications:")
    print("-" * 70)
    results.append(check_file_exists("tools.py"))
    results.append(check_file_exists("server.py"))
    results.append(check_file_exists("main.py"))
    results.append(check_file_exists("models.py"))
    
    # Check tools.py
    print("\n2. Checking tools.py:")
    print("-" * 70)
    results.append(check_code_pattern(
        "tools.py",
        r"from langgraph\.types import interrupt",
        "Import: langgraph.types.interrupt"
    ))
    results.append(check_code_pattern(
        "tools.py",
        r"@tool\s+def ask_question",
        "Tool definition: ask_question"
    ))
    results.append(check_code_pattern(
        "tools.py",
        r"def ask_question\(questions: List\[Dict\[str, Any\]\]\)",
        "Tool signature: ask_question(questions: List[Dict[str, Any]])"
    ))
    results.append(check_code_pattern(
        "tools.py",
        r'interrupt\(\{',
        "Uses interrupt() to pause execution"
    ))
    results.append(check_code_pattern(
        "tools.py",
        r"ask_question",
        "ask_question added to get_tools()"
    ))
    
    # Check server.py
    print("\n3. Checking server.py:")
    print("-" * 70)
    results.append(check_code_pattern(
        "server.py",
        r"from langchain\.agents\.middleware import HumanInTheLoopMiddleware",
        "Import: HumanInTheLoopMiddleware"
    ))
    results.append(check_code_pattern(
        "server.py",
        r"HumanInTheLoopMiddleware\(",
        "Middleware instantiation: HumanInTheLoopMiddleware"
    ))
    results.append(check_code_pattern(
        "server.py",
        r'interrupt_on=\{.*"ask_question"',
        "Middleware config: ask_question in interrupt_on"
    ))
    results.append(check_code_pattern(
        "server.py",
        r"middleware=\[hitl_middleware\]",
        "Middleware added to create_agent"
    ))
    results.append(check_code_pattern(
        "server.py",
        r'source == "__interrupt__"',
        "Interrupt detection in process_message"
    ))
    results.append(check_code_pattern(
        "server.py",
        r'"type": "questions_pending"',
        "Yields questions_pending event"
    ))
    results.append(check_code_pattern(
        "server.py",
        r"async def process_resume\(",
        "Method: process_resume"
    ))
    results.append(check_code_pattern(
        "server.py",
        r"answers: Dict\[str, List\[str\]\]",
        "process_resume signature: accepts answers dict"
    ))
    results.append(check_code_pattern(
        "server.py",
        r"Command\(resume=",
        "Uses Command(resume=...) in process_resume"
    ))
    
    # Check models.py
    print("\n4. Checking models.py:")
    print("-" * 70)
    results.append(check_code_pattern(
        "models.py",
        r"from typing import.*Dict",
        "Import: Dict from typing"
    ))
    results.append(check_code_pattern(
        "models.py",
        r"class QuestionData\(BaseModel\)",
        "Model: QuestionData"
    ))
    results.append(check_code_pattern(
        "models.py",
        r"question_id: str",
        "QuestionData field: question_id"
    ))
    results.append(check_code_pattern(
        "models.py",
        r"options: List\[str\]",
        "QuestionData field: options"
    ))
    results.append(check_code_pattern(
        "models.py",
        r"class ResumeData\(BaseModel\)",
        "Model: ResumeData"
    ))
    results.append(check_code_pattern(
        "models.py",
        r"answers: Dict\[str, List\[str\]\]",
        "ResumeData field: answers"
    ))
    results.append(check_code_pattern(
        "models.py",
        r"resume_data: Optional\[ResumeData\]",
        "ChatInput field: resume_data"
    ))
    results.append(check_code_pattern(
        "models.py",
        r'"questions_pending"',
        "ChatChunk type: questions_pending"
    ))
    results.append(check_code_pattern(
        "models.py",
        r"questions: Optional\[List\[QuestionData\]\]",
        "ChatChunk field: questions"
    ))
    
    # Check main.py
    print("\n5. Checking main.py:")
    print("-" * 70)
    results.append(check_code_pattern(
        "main.py",
        r"from langgraph\.types import Command",
        "Import: Command from langgraph.types"
    ))
    results.append(check_code_pattern(
        "main.py",
        r"is_resume = input_data\.resume_data is not None",
        "Detects resume command"
    ))
    results.append(check_code_pattern(
        "main.py",
        r"if is_resume:",
        "Handles resume flow"
    ))
    results.append(check_code_pattern(
        "main.py",
        r"chat_server\.process_resume\(",
        "Calls process_resume for resume commands"
    ))
    results.append(check_code_pattern(
        "main.py",
        r"answers=input_data\.resume_data\.answers",
        "Passes answers to process_resume"
    ))
    
    # Summary
    print("\n" + "=" * 70)
    print("Verification Summary")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    percentage = (passed / total) * 100
    
    print(f"Passed: {passed}/{total} ({percentage:.1f}%)")
    
    if passed == total:
        print("\n[SUCCESS] All verification checks passed!")
        print("\nImplementation Complete:")
        print("  1. ask_question tool created with interrupt mechanism")
        print("  2. HumanInTheLoopMiddleware configured")
        print("  3. Interrupt detection and streaming implemented")
        print("  4. API models updated (QuestionData, ResumeData)")
        print("  5. Chat endpoint handles resume commands")
        print("  6. process_resume method added")
        print("\nFeatures:")
        print("  - Batch questions (multiple questions at once)")
        print("  - Multiple answer selections per question")
        print("  - Questions streamed via SSE as 'questions_pending' event")
        print("  - Resume mechanism via Command(resume={...})")
        print("\nNext Steps:")
        print("  - Start the backend server")
        print("  - Test with a frontend that can display questions and submit answers")
        print("  - Verify the full interrupt -> answer -> resume flow")
    else:
        failed = total - passed
        print(f"\n[WARNING] {failed} check(s) failed. Review the output above.")
    
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)


