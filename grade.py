import json
from pathlib import Path
from typing import Dict
from tqdm import tqdm
from llm_client import LLMClient, Task, Request

GRADING_PROMPT = """
**Problem:**
{problem}

**Ground Truth Answer:**
{ground_truth}

**Model's Answer:**
{model_answer}

**Task: Grading**
Please determine if the model's answer is correct compared to the ground truth answer.

**Guidelines:**
- Consider mathematical equivalence (e.g., 1/2 = 0.5, 2x = x + x)
- Ignore formatting differences if the mathematical content is the same
- Answer with \\boxed{{YES}} if correct, or \\boxed{{NO}} if incorrect
"""

MAX_WORKERS = 100
MAX_TRY = 10


class AnswerGrader:
    def __init__(self, judge_model_type='deepseek', target='MATH500'):
        self.client = LLMClient()
        self.judge_model_type = judge_model_type
        self.target = target

    def get_problem_and_solution(self, item: Dict) -> tuple:
        """Extract problem and solution based on target dataset"""
        if self.target == 'MATH500':
            problem = item['problem']
            solution = item['solution']
        elif self.target.startswith('AIME2025'):
            problem = item['question']
            solution = item['answer']
        else:
            raise ValueError(f"Unsupported target: {self.target}")

        return problem, solution

    def _create_grading_tasks(self, data):
        """Create grading tasks from data"""
        tasks = []
        for i, item in enumerate(data):
            if 'result' not in item:
                continue

            problem, ground_truth = self.get_problem_and_solution(item)
            model_answer = item['result']['answer']

            task = Task(
                index=i,
                request=Request(
                    query=GRADING_PROMPT.format(
                        problem=problem,
                        ground_truth=ground_truth,
                        model_answer=model_answer
                    ),
                    model_type=self.judge_model_type,
                    system_prompt="You are a helpful assistant",
                    reasoning_on=False
                ),
                metadata={'item': item}
            )
            tasks.append(task)
        return tasks

    def _process_response(self, task):
        """Process a single grading response"""
        if not task.response.success:
            raise Exception(f"Generation failed for item {task.index}")

        item = task.metadata['item']

        include_yes = 'YES' in task.response.content
        include_no = 'NO' in task.response.content
        if include_yes and not include_no:
            is_correct = True
        elif (include_yes and include_no) or (not include_yes and not include_no):
            raise Exception
        else:
            is_correct = False

        return {
            'index': task.index,
            'unique_id': item.get('unique_id', f'item_{task.index}'),
            'correct': is_correct,
            'grading_cot': task.response.content
        }

    def _save_results(self, output_path, grades, summary=None):
        """Save grading results to file"""
        output_data = {
            'summary': summary,
            'grades': sorted(grades, key=lambda x: x['index'])
        }
        with open(output_path, 'w') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    def grade_results(self, results_path: str, output_path: str):
        """Grade all items in results.json and save to grades.json"""
        # Load results
        with open(results_path, 'r') as f:
            data = json.load(f)

        # Create tasks for concurrent grading
        tasks = self._create_grading_tasks(data)

        # Grade concurrently with retries
        grades = []
        correct_count = 0
        retry_queue = tasks

        for attempt in range(MAX_TRY):
            if not retry_queue:
                break

            desc = f"Grading answers (attempt {attempt + 1}/{MAX_TRY})"
            next_retry_queue = []

            for task in tqdm(self.client.generate_concurrent(retry_queue, max_workers=MAX_WORKERS),
                               total=len(retry_queue), desc=desc):
                try:
                    grade_entry = self._process_response(task)

                    if grade_entry['correct']:
                        correct_count += 1

                    grades.append(grade_entry)

                    # Save grades incrementally
                    self._save_results(output_path, grades)
                except Exception as e:
                    if attempt < MAX_TRY - 1:
                        print(f"\nError processing task {task.index}, will retry: {e}")
                        next_retry_queue.append(task)
                    else:
                        print(f"\nTask {task.index} failed after {MAX_TRY} attempts: {e}")

            retry_queue = next_retry_queue

        # Final save with complete summary
        total = len(grades)
        accuracy = (correct_count / total * 100) if total > 0 else 0

        summary = {
            'total': total,
            'correct': correct_count,
            'incorrect': total - correct_count,
            'accuracy': round(accuracy, 2)
        }

        self._save_results(output_path, grades, summary)

        print(f"\nGrading complete. Results saved to {output_path}")
        print(f"Accuracy: {correct_count}/{total} ({accuracy:.2f}%)")


if __name__ == '__main__':
    import sys

    # Default parameters
    results_path = 'data/MATH500/deepseek/p1/results.json'
    judge_model_type = 'deepseek'
    target = 'MATH500'

    # Allow command line override
    assert len(sys.argv) == 1 + 0 or len(sys.argv) == 1 + 1 or len(sys.argv) == 1 + 2 or len(sys.argv) == 1 + 3
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    if len(sys.argv) > 2:
        judge_model_type = sys.argv[2]
    if len(sys.argv) > 3:
        target = sys.argv[3]

    # Determine output path (same directory as results.json)
    output_path = str(Path(results_path).parent / 'grades.json')

    grader = AnswerGrader(judge_model_type=judge_model_type, target=target)
    grader.grade_results(results_path, output_path)
