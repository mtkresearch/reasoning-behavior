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

MAX_WORKERS = 300


class AnswerGrader:
    def __init__(self, model_type='deepseek'):
        self.client = LLMClient()
        self.model_type = model_type

    def get_problem_and_solution(self, item: Dict) -> tuple:
        """Extract problem and solution based on dataset/model_type"""
        if self.model_type == 'deepseek':
            problem = item.get('problem', '')
            solution = item.get('solution', '')
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        return problem, solution

    def _create_grading_tasks(self, items_to_grade):
        """Create grading tasks from items"""
        tasks = []
        for i, item in items_to_grade:
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
                    model_type='deepseek',
                    system_prompt="You are a helpful assistant",
                    reasoning_on=False
                ),
                metadata={'unique_id': item['unique_id']}
            )
            tasks.append(task)
        return tasks

    def _process_response(self, task):
        """Process a single grading response"""
        if not task.response.success:
            raise Exception(f"Generation failed for item {task.index}")

        try:
            is_correct = '\\boxed{YES}' in task.response.content
            return {
                'index': task.index,
                'unique_id': task.metadata['unique_id'],
                'correct': is_correct,
                'grading_cot': task.response.content
            }
        except Exception as e:
            print(f"\nError processing response for item {task.index}: {e}")
            return {
                'index': task.index,
                'unique_id': task.metadata['unique_id'],
                'correct': None,
                'grading_cot': None,
                'error': str(e)
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

        # Filter items that have results
        items_to_grade = [(i, item) for i, item in enumerate(data) if 'result' in item]
        print(f"Grading {len(items_to_grade)} items...")

        # Create tasks for concurrent grading
        tasks = self._create_grading_tasks(items_to_grade)

        # Grade concurrently
        grades = []
        correct_count = 0

        for task in tqdm(self.client.generate_concurrent(tasks, max_workers=MAX_WORKERS),
                           total=len(tasks), desc="Grading answers"):
            grade_entry = self._process_response(task)

            if grade_entry['correct']:
                correct_count += 1

            grades.append(grade_entry)

            # Save grades incrementally
            self._save_results(output_path, grades)

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

    # Default paths
    results_path = 'data/MATH500/deepseek/p1/results.json'
    model_type = 'deepseek'  # The model type of the results being graded

    # Allow command line override
    assert len(sys.argv) == 1 + 0 or len(sys.argv) == 1 + 2
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    if len(sys.argv) > 2:
        model_type = sys.argv[2]

    # Determine output path (same directory as results.json)
    output_path = str(Path(results_path).parent / 'grades.json')

    grader = AnswerGrader(model_type=model_type)
    grader.grade_results(results_path, output_path)
