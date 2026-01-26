"""
Test script for the Chat API endpoint.
Tests all provided questions and compares responses with expected answers.
Loads test cases from test_questions.json file.
"""

import requests
import json
import time
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from pydantic import BaseModel, Field
from openai import OpenAI
import dotenv

dotenv.load_dotenv("secrets.env")


class TestEvaluation(BaseModel):
    """Pydantic model for structured test evaluation output."""
    passed: bool = Field(description="Whether the test passed (score >= 70)")
    score: float = Field(ge=0, le=100, description="Score from 0-100 indicating how well the actual answer matches the expected answer")
    reasoning: str = Field(description="Detailed explanation of the evaluation, including what matched and what didn't")
    key_findings: List[str] = Field(description="List of key findings, such as missing information, incorrect values, or strengths")


@dataclass
class TestCase:
    """Represents a test case with question and expected answer."""
    question: str
    expected_answer: str
    source: str
    rationale: Optional[str] = None
    complexity: str = "Simple"
    id: Optional[int] = None
    category: Optional[str] = None


def load_test_cases(json_path: str = "test_questions.json", complexity_filter: Optional[str] = None) -> List[TestCase]:
    """
    Load test cases from JSON file.
    
    Args:
        json_path: Path to the JSON file containing test cases
        complexity_filter: Optional complexity level to filter by (Simple, Medium, Complex, Extremely Complex)
    
    Returns:
        List of TestCase objects
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(script_dir, json_path)
    
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Test questions file not found: {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_cases = []
    for case_data in data.get("test_cases", []):
        # Apply complexity filter if specified
        if complexity_filter and case_data.get("complexity") != complexity_filter:
            continue
        
        test_case = TestCase(
            question=case_data.get("question", ""),
            expected_answer=case_data.get("expected_answer", ""),
            source=case_data.get("source", ""),
            rationale=case_data.get("rationale"),
            complexity=case_data.get("complexity", "Simple"),
            id=case_data.get("id"),
            category=case_data.get("category")
        )
        test_cases.append(test_case)
    
    return test_cases


class ChatAPITester:
    """Test harness for the Chat API endpoint."""
    
    def __init__(
        self,
        base_url: str = "https://fmcg-agent.elevatics.site",
        user_id: str = "test_user@gmail.com",
        judge_api_key: Optional[str] = None,
        judge_model: str = "google/gemini-3-flash-preview"
    ):
        self.base_url = base_url.rstrip('/')
        self.user_id = user_id
        self.chat_endpoint = f"{self.base_url}/api/v1/chat"
        self.results: List[Dict] = []
        self.judge_api_key = judge_api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.judge_model = judge_model
        
        # Initialize OpenAI client for judge
        self.judge_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.judge_api_key,
        ) if self.judge_api_key else None
    
    def test_health(self) -> bool:
        """Test if the API is healthy."""
        try:
            response = requests.get(f"{self.base_url}/api/v1/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def send_chat_message(
        self,
        query: str,
        thread_id: Optional[str] = None,
        model_id: Optional[str] = None
    ) -> Dict:
        """
        Send a chat message and collect the streaming response.
        
        Returns:
            Dictionary with full_response content and tool events
        """
        payload = {
            "query": query,
            "user_id": self.user_id,
        }
        
        if thread_id:
            payload["thread_id"] = thread_id
        if model_id:
            payload["model_id"] = model_id
        
        try:
            response = requests.post(
                self.chat_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                stream=True,
                timeout=120  # 2 minute timeout for complex queries
            )
            
            if response.status_code != 200:
                return {
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "content": "",
                    "tool_events": []
                }
            
            # Parse SSE stream
            full_content = ""
            tool_events = []
            current_tool = None
            
            for line in response.iter_lines():
                if not line:
                    continue
                
                # SSE format: data: {...}
                if line.startswith(b'data: '):
                    data_str = line[6:].decode('utf-8')
                    try:
                        chunk = json.loads(data_str)
                        chunk_type = chunk.get("type", "")
                        
                        if chunk_type == "chunk":
                            full_content += chunk.get("content", "")
                        elif chunk_type == "tool_start":
                            current_tool = {
                                "name": chunk.get("name"),
                                "input": chunk.get("input"),
                                "start_time": time.time()
                            }
                        elif chunk_type == "tool_end":
                            if current_tool:
                                current_tool["output"] = chunk.get("output", "")
                                current_tool["end_time"] = time.time()
                                tool_events.append(current_tool)
                                current_tool = None
                        elif chunk_type == "full_response":
                            # Final response - use this content if available
                            if chunk.get("content"):
                                full_content = chunk.get("content", "")
                        elif chunk_type == "error":
                            return {
                                "error": chunk.get("content", "Unknown error"),
                                "content": full_content,
                                "tool_events": tool_events
                            }
                    except json.JSONDecodeError:
                        continue
            
            return {
                "content": full_content,
                "tool_events": tool_events,
                "error": None
            }
            
        except requests.exceptions.Timeout:
            return {
                "error": "Request timeout (120s exceeded)",
                "content": "",
                "tool_events": []
            }
        except Exception as e:
            return {
                "error": str(e),
                "content": "",
                "tool_events": []
            }
    
    def judge_test_result(
        self,
        question: str,
        expected_answer: str,
        actual_answer: str,
        rationale: Optional[str] = None
    ) -> Dict:
        """
        Use an LLM judge with structured outputs to evaluate test results.
        
        Returns:
            Dictionary with passed (bool), score (0-100), reasoning (str), and key_findings (list)
        """
        if not self.judge_client:
            # Fallback if no API key or client
            return {
                "passed": False,
                "score": 0,
                "reasoning": "No judge API key provided, falling back to heuristic evaluation",
                "key_findings": []
            }
        
        # Build the prompt for the judge
        judge_prompt = f"""You are an expert test evaluator. Your task is to determine if an actual answer correctly addresses the expected answer for a given question.

Question: {question}

Expected Answer: {expected_answer}
{f'Rationale: {rationale}' if rationale else ''}

Actual Answer: {actual_answer}

Evaluate whether the actual answer correctly addresses the question and matches the expected answer. Consider:
1. Does the actual answer contain the key information from the expected answer?
2. Are the numbers, values, or facts correct?
3. Is the answer contextually appropriate?
4. Are there any significant errors or omissions?

Provide a score from 0-100 where:
- 90-100: Excellent match, all key information present and correct
- 70-89: Good match, most key information present with minor issues
- 50-69: Partial match, some key information missing or incorrect
- 0-49: Poor match, significant errors or missing critical information
"""
        
        try:
            response = self.judge_client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that evaluates test results in JSON format."},
                    {"role": "user", "content": judge_prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "test_evaluation_schema",
                        "strict": True,
                        "schema": TestEvaluation.model_json_schema()
                    }
                }
            )
            
            # Parse the structured content
            content = response.choices[0].message.content
            evaluation = json.loads(content)
            
            return {
                "passed": evaluation.get("passed", False),
                "score": evaluation.get("score", 0),
                "reasoning": evaluation.get("reasoning", ""),
                "key_findings": evaluation.get("key_findings", [])
            }
            
        except json.JSONDecodeError as e:
            return {
                "passed": False,
                "score": 0,
                "reasoning": f"Failed to parse judge response: {e}",
                "key_findings": []
            }
        except Exception as e:
            return {
                "passed": False,
                "score": 0,
                "reasoning": f"Judge evaluation error: {str(e)}",
                "key_findings": []
            }
    
    def run_test_case(self, test_case: TestCase, test_number: int) -> Dict:
        """Run a single test case and return results."""
        print(f"\n{'='*80}")
        print(f"Test {test_number}: {test_case.complexity} Complexity")
        print(f"{'='*80}")
        print(f"Question: {test_case.question}")
        print(f"Expected Answer: {test_case.expected_answer}")
        if test_case.rationale:
            print(f"Rationale: {test_case.rationale}")
        print(f"Source: {test_case.source}")
        print(f"\nSending request to API...")
        
        start_time = time.time()
        response = self.send_chat_message(test_case.question)
        elapsed_time = time.time() - start_time
        
        result = {
            "test_number": test_number,
            "question": test_case.question,
            "expected_answer": test_case.expected_answer,
            "actual_answer": response.get("content", ""),
            "error": response.get("error"),
            "tool_events": response.get("tool_events", []),
            "elapsed_time": elapsed_time,
            "source": test_case.source,
            "rationale": test_case.rationale,
            "complexity": test_case.complexity,
            "passed": False,
            "judge_score": 0,
            "judge_reasoning": "",
            "judge_findings": []
        }
        
        if response.get("error"):
            print(f"❌ Error: {response['error']}")
            result["judge_reasoning"] = f"Test failed due to API error: {response['error']}"
        else:
            print(f"\n✅ Response received ({elapsed_time:.2f}s)")
            print(f"Actual Answer: {result['actual_answer'][:500]}...")  # Truncate for display
            print(f"Tool Events: {len(result['tool_events'])}")
            if result["tool_events"]:
                for tool in result["tool_events"]:
                    print(f"  - {tool['name']}: {tool.get('output', '')[:100]}...")
            
            # Use LLM judge to evaluate the result
            print(f"\n⚖️  Evaluating with LLM judge...")
            judge_result = self.judge_test_result(
                question=test_case.question,
                expected_answer=test_case.expected_answer,
                actual_answer=result["actual_answer"],
                rationale=test_case.rationale
            )
            
            result["passed"] = judge_result["passed"]
            result["judge_score"] = judge_result["score"]
            result["judge_reasoning"] = judge_result["reasoning"]
            result["judge_findings"] = judge_result["key_findings"]
            
            print(f"Judge Score: {judge_result['score']}/100")
            print(f"Judge Reasoning: {judge_result['reasoning'][:200]}...")
            if judge_result["key_findings"]:
                print(f"Key Findings:")
                for finding in judge_result["key_findings"][:3]:  # Show first 3
                    print(f"  - {finding}")
            print(f"Status: {'✅ PASSED' if result['passed'] else '❌ FAILED'}")
        
        return result
    
    def run_all_tests(self, test_cases: List[TestCase], output_dir: str = "test_results", output_file: Optional[str] = None) -> Dict:
        """Run all test cases and generate a summary report."""
        print("="*80)
        print("CHAT API TEST SUITE")
        print("="*80)
        
        # Health check
        print("\n🔍 Checking API health...")
        if not self.test_health():
            print("❌ API is not healthy. Please ensure the server is running.")
            return {
                "success": False,
                "message": "API health check failed",
                "results": []
            }
        print("✅ API is healthy")
        
        # Run all test cases
        print(f"\n📋 Running {len(test_cases)} test cases...")
        for i, test_case in enumerate(test_cases, 1):
            test_number = test_case.id if test_case.id else i
            result = self.run_test_case(test_case, test_number)
            self.results.append(result)
            
            # Save incrementally after each test
            self.save_results(output_dir=output_dir, filename=output_file, show_message=False)
            print(f"💾 Progress saved ({i}/{len(test_cases)} tests completed)")
            
            time.sleep(1)  # Small delay between requests
        
        # Generate and save final summary
        summary = self.generate_summary()
        self.save_results(output_dir=output_dir, filename=output_file, show_message=True)
        return summary
    
    def generate_summary(self) -> Dict:
        """Generate a summary report of all test results."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed
        errors = sum(1 for r in self.results if r.get("error"))
        
        # Group by complexity
        complexity_levels = ["Simple", "Medium", "Complex", "Extremely Complex"]
        by_complexity = {}
        
        for complexity in complexity_levels:
            passed_count = sum(1 for r in self.results if r["complexity"] == complexity and r["passed"])
            total_count = sum(1 for r in self.results if r["complexity"] == complexity)
            if total_count > 0:
                by_complexity[complexity.lower().replace(" ", "_")] = {
                    "passed": passed_count,
                    "total": total_count,
                    "pass_rate": (passed_count / total_count * 100) if total_count > 0 else 0
                }
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": (passed / total * 100) if total > 0 else 0,
            "by_complexity": by_complexity,
            "results": self.results
        }
        
        # Print summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️  Errors: {errors}")
        print(f"📊 Pass Rate: {summary['pass_rate']:.1f}%")
        print(f"\nBy Complexity:")
        for complexity in complexity_levels:
            key = complexity.lower().replace(" ", "_")
            if key in by_complexity:
                stats = by_complexity[key]
                print(f"  {complexity}: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.1f}%)")
        print("="*80)
        
        return summary
    
    def save_results(self, output_dir: str = "test_results", filename: Optional[str] = None, show_message: bool = True):
        """
        Save test results to a JSON file with timestamp.
        
        Args:
            output_dir: Directory to save results in (default: test_results)
            filename: Optional custom filename. If not provided, uses timestamp.
            show_message: Whether to print save confirmation message (default: True)
        """
        # Create output directory if it doesn't exist
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, output_dir)
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename with timestamp if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"
        
        # Store the filepath for reuse in incremental saves
        if not hasattr(self, '_output_filepath'):
            self._output_filepath = os.path.join(results_dir, filename)
        
        # Generate comprehensive summary (without printing)
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed
        errors = sum(1 for r in self.results if r.get("error"))
        
        # Group by complexity
        complexity_levels = ["Simple", "Medium", "Complex", "Extremely Complex"]
        by_complexity = {}
        
        for complexity in complexity_levels:
            passed_count = sum(1 for r in self.results if r["complexity"] == complexity and r["passed"])
            total_count = sum(1 for r in self.results if r["complexity"] == complexity)
            if total_count > 0:
                by_complexity[complexity.lower().replace(" ", "_")] = {
                    "passed": passed_count,
                    "total": total_count,
                    "pass_rate": (passed_count / total_count * 100) if total_count > 0 else 0
                }
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": (passed / total * 100) if total > 0 else 0,
            "by_complexity": by_complexity,
            "results": self.results
        }
        
        # Save to file
        with open(self._output_filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        if show_message:
            print(f"\n💾 Results saved to {self._output_filepath}")


def main():
    """Main entry point for the test script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the Chat API endpoint")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--user-id",
        default="test_user@gmail.com",
        help="User ID for testing (default: test_user@gmail.com)"
    )
    parser.add_argument(
        "--output-dir",
        default="test_results",
        help="Output directory for results (default: test_results)"
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output filename (default: auto-generated with timestamp)"
    )
    parser.add_argument(
        "--test-number",
        type=int,
        help="Run only a specific test number (by ID from JSON)"
    )
    parser.add_argument(
        "--complexity",
        choices=["Simple", "Medium", "Complex", "Extremely Complex"],
        help="Filter tests by complexity level"
    )
    parser.add_argument(
        "--json-file",
        default="test_questions.json",
        help="Path to JSON file containing test cases (default: test_questions.json)"
    )
    parser.add_argument(
        "--judge-api-key",
        default=None,
        help="OpenRouter API key for LLM judge (default: from OPENROUTER_API_KEY env var)"
    )
    parser.add_argument(
        "--judge-model",
        default="google/gemini-3-flash-preview",
        help="Model to use for LLM judge (default: google/gemini-3-flash-preview)"
    )
    
    args = parser.parse_args()
    
    # Load test cases from JSON
    try:
        test_cases = load_test_cases(args.json_file, args.complexity)
        if not test_cases:
            print(f"❌ No test cases found")
            if args.complexity:
                print(f"   (with complexity filter: {args.complexity})")
            return
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    except Exception as e:
        print(f"❌ Error loading test cases: {e}")
        return
    
    tester = ChatAPITester(
        base_url=args.url,
        user_id=args.user_id,
        judge_api_key=args.judge_api_key,
        judge_model=args.judge_model
    )
    
    if args.test_number:
        # Run single test by ID
        matching_tests = [tc for tc in test_cases if tc.id == args.test_number]
        if matching_tests:
            test_case = matching_tests[0]
            result = tester.run_test_case(test_case, args.test_number)
            tester.results = [result]
            tester.save_results(output_dir=args.output_dir, filename=args.output_file, show_message=True)
        else:
            print(f"❌ Test with ID {args.test_number} not found")
            if args.complexity:
                print(f"   (Note: complexity filter '{args.complexity}' is active)")
    else:
        # Run all tests (or filtered by complexity)
        if args.complexity:
            print(f"🔍 Filtering tests by complexity: {args.complexity}")
        tester.run_all_tests(test_cases, output_dir=args.output_dir, output_file=args.output_file)


if __name__ == "__main__":
    main()

