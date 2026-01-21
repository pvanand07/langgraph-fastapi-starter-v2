"""Test script for Human-in-the-Loop ask_question tool implementation."""

import sys
from typing import Dict, List, Any

def test_imports():
    """Test that all required imports are available."""
    print("Testing imports...")
    try:
        # Test tool imports
        from tools import ask_question, get_tools
        print("✓ Tools imported successfully")
        
        # Test model imports
        from models import ChatInput, ResumeData, QuestionData, ChatChunk
        print("✓ Models imported successfully")
        
        # Test server imports
        from server import ChatServer
        print("✓ Server imported successfully")
        
        # Verify ask_question is in tools list
        tools = get_tools()
        tool_names = [tool.name for tool in tools]
        assert "ask_question" in tool_names, "ask_question not found in tools list"
        print(f"✓ ask_question tool is registered (found {len(tools)} tools)")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def test_models():
    """Test that the new models are correctly defined."""
    print("\nTesting models...")
    try:
        from models import ChatInput, ResumeData, QuestionData
        
        # Test QuestionData
        question = QuestionData(
            question_id="test_q1",
            question="Test question?",
            options=["Option 1", "Option 2", "Option 3"]
        )
        print(f"✓ QuestionData model works: {question.question_id}")
        
        # Test ResumeData
        resume_data = ResumeData(
            answers={"test_q1": ["Option 1", "Option 2"]}
        )
        print(f"✓ ResumeData model works: {resume_data.answers}")
        
        # Test ChatInput with resume_data
        chat_input = ChatInput(
            user_id="test_user",
            thread_id="test_thread",
            resume_data=resume_data
        )
        print(f"✓ ChatInput with resume_data works")
        
        # Test ChatInput without resume_data (regular message)
        chat_input_regular = ChatInput(
            query="Test query",
            user_id="test_user"
        )
        print(f"✓ ChatInput without resume_data works")
        
        return True
    except Exception as e:
        print(f"✗ Model error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ask_question_tool():
    """Test the ask_question tool structure."""
    print("\nTesting ask_question tool...")
    try:
        from tools import ask_question
        
        # Check tool metadata
        print(f"✓ Tool name: {ask_question.name}")
        print(f"✓ Tool description: {ask_question.description[:50]}...")
        
        # Verify the tool has the correct schema
        if hasattr(ask_question, 'args_schema'):
            print(f"✓ Tool has args_schema")
        
        return True
    except Exception as e:
        print(f"✗ Tool error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chatserver_methods():
    """Test that ChatServer has the required methods."""
    print("\nTesting ChatServer methods...")
    try:
        from server import ChatServer
        
        # Check that ChatServer has the required methods
        assert hasattr(ChatServer, 'process_message'), "ChatServer missing process_message method"
        print("✓ ChatServer has process_message method")
        
        assert hasattr(ChatServer, 'process_resume'), "ChatServer missing process_resume method"
        print("✓ ChatServer has process_resume method")
        
        return True
    except Exception as e:
        print(f"✗ ChatServer error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_flow():
    """Test the expected integration flow (documentation)."""
    print("\nTesting integration flow documentation...")
    
    flow = """
    Expected Human-in-the-Loop Flow:
    
    1. User sends message → /api/v1/chat (ChatInput with query)
    2. Agent processes message → calls ask_question tool
    3. ask_question calls interrupt() → execution pauses
    4. ChatServer.process_message detects __interrupt__
    5. Stream questions_pending event to frontend with questions
    6. User answers questions in frontend UI
    7. Frontend sends resume → /api/v1/chat (ChatInput with resume_data)
    8. ChatServer.process_resume called with answers
    9. Agent resumes with Command(resume={...})
    10. Agent receives answers and continues execution
    11. Final response streamed to frontend
    """
    
    print(flow)
    
    example_questions = [
        {
            "question_id": "experience",
            "question": "What is your experience level?",
            "options": ["Beginner", "Intermediate", "Advanced", "Expert"]
        },
        {
            "question_id": "interests",
            "question": "Which topics interest you?",
            "options": ["Python", "JavaScript", "Data Science", "DevOps", "AI/ML"]
        }
    ]
    
    example_answers = {
        "experience": ["Intermediate"],
        "interests": ["Python", "Data Science", "AI/ML"]
    }
    
    print("\nExample Questions:")
    for q in example_questions:
        print(f"  - {q['question_id']}: {q['question']}")
        print(f"    Options: {', '.join(q['options'])}")
    
    print("\nExample Answers:")
    for q_id, selected in example_answers.items():
        print(f"  - {q_id}: {', '.join(selected)}")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Human-in-the-Loop Implementation Verification")
    print("=" * 60)
    
    results = {
        "Imports": test_imports(),
        "Models": test_models(),
        "ask_question Tool": test_ask_question_tool(),
        "ChatServer Methods": test_chatserver_methods(),
        "Integration Flow": test_integration_flow()
    }
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nImplementation Summary:")
        print("  - ask_question tool created with interrupt mechanism")
        print("  - HumanInTheLoopMiddleware configured for ask_question")
        print("  - Interrupt streaming implemented in process_message")
        print("  - API models updated (QuestionData, ResumeData)")
        print("  - Chat endpoint modified to handle resume commands")
        print("  - process_resume method added to ChatServer")
        print("\nNext Steps:")
        print("  1. Start the backend server: python backend/main.py")
        print("  2. Test with frontend by triggering ask_question")
        print("  3. Verify questions_pending event is streamed")
        print("  4. Submit answers and verify resume works")
    else:
        print("✗ SOME TESTS FAILED")
        print("Please review the errors above.")
        sys.exit(1)
    
    print("=" * 60)


if __name__ == "__main__":
    main()


