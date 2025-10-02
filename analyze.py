import json
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm
from llm_client import LLMClient, Task, Request

DIRECT_REASONING_WAY_SELECTION = """
** Original Reasoning Chain **:
```markdown
{traj}
```

**Task: Reasoning Strategy Determination**
Common reasoning strategies used in problem solving are listed below:
A). Planning and Execution: Plan first, then execute.
B). Problem Decomposition: Break down the problem into smaller parts and tackle each one individually.
C). Formulate Hypotheses and Test Them: Propose hypotheses before proceeding with verification.
D). Guess the Answer First, Then Verify: Make an initial guess, then check if it's correct.
E). Option Judgment and Elimination: List all possible options and eliminate them one by one.
F). Reverse Thinking: Work backwards from the result to infer the cause.
G). Divergent and Convergent Thinking: Use brainstorming for divergent thinking first, then converge to form a solution.
H). Counterexample Testing: Imagine scenarios where the hypothesis holds, then look for counterexamples to disprove it.
I). Association Method: Draw connections or analogies from related concepts, experiences, or fields to inspire solutions.

Please Determine: Does the reasoning chain use the {option}) strategy or not?

**Guidelines:**
- THINK IT STEP BY STEP First and Answer with \\boxed{{YES}} or \\boxed{{NO}}
- You need to show your evidences before answering.
- Detailed operation steps do not count as strategy.
"""

MAX_WORKERS = 9
MAX_TRY = 3

REASONING_STRATEGIES = {
    'A': 'Planning and Execution',
    'B': 'Problem Decomposition',
    'C': 'Formulate Hypotheses and Test Them',
    'D': 'Guess the Answer First, Then Verify',
    'E': 'Option Judgment and Elimination',
    'F': 'Reverse Thinking',
    'G': 'Divergent and Convergent Thinking',
    'H': 'Counterexample Testing',
    'I': 'Association Method'
}


class ReasoningAnalyzer:
    def __init__(self):
        self.client = LLMClient()

    def analyze_single_item(self, item: Dict, index: int) -> Dict:
        """Analyze a single problem's reasoning trajectory using concurrent strategy checking"""
        traj = item['result']['traj']

        # Create tasks for all strategies
        tasks = []
        for option, strategy_name in REASONING_STRATEGIES.items():
            task = Task(
                index=ord(option) - ord('A'),  # Convert A-I to 0-8
                request=Request(
                    query=DIRECT_REASONING_WAY_SELECTION.format(traj=traj, option=option),
                    model_type='deepseek',
                    system_prompt="You are a helpful assistant",
                    extra_body={"chat_template_kwargs": {"thinking": False}}
                ),
                metadata={'option': option, 'strategy_name': strategy_name}
            )
            tasks.append(task)

        strategies_found = []
        cot_collection = {}

        # Process all strategies concurrently
        for completed_task in self.client.generate_concurrent(tasks, max_workers=MAX_WORKERS):
            option = completed_task.metadata['option']
            strategy_name = completed_task.metadata['strategy_name']

            if not completed_task.response.success:
                raise Exception(f"Generation failed for strategy {option} in item {index}")

            try:
                has_strategy = '\\boxed{YES}' in completed_task.response.content
                cot_collection[option] = {
                    'name': strategy_name,
                    'response': completed_task.response.content,
                    'found': has_strategy
                }
                if has_strategy:
                    strategies_found.append(option)
            except Exception as e:
                print(f"Error processing strategy {option} for item {index}: {e}")
                cot_collection[option] = {
                    'name': strategy_name,
                    'error': str(e),
                    'found': False
                }

        return {
            'strategies_found': sorted(strategies_found),
            'cot_collection': cot_collection
        }

    def analyze_results(self, results_path: str, output_path: str):
        """Analyze all items in results.json and save to metrics.json"""
        # Load results
        with open(results_path, 'r') as f:
            data = json.load(f)

        # Filter items that have results
        items_to_analyze = [(i, item) for i, item in enumerate(data) if 'result' in item]

        print(f"Analyzing {len(items_to_analyze)} items...")

        metrics = []
        for i, item in tqdm(items_to_analyze, desc="Analyzing reasoning strategies"):
            analysis = self.analyze_single_item(item, i)

            metrics.append({
                'index': i,
                'unique_id': item.get('unique_id', ''),
                'strategies_found': analysis['strategies_found'],
                'cot_collection': analysis['cot_collection']
            })

            # Save metrics incrementally
            output_data = {
                'summary': None,
                'metrics': metrics
            }
            with open(output_path, 'w') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

        summary = self.calculate_summary(metrics)
        output_data = {
            'summary': summary,
            'metrics': metrics
        }
        with open(output_path, 'w') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Analysis complete. Metrics saved to {output_path}")

        # Print summary
        self.print_summary(summary, len(metrics))

    def calculate_summary(self, metrics: List[Dict]) -> Dict:
        """Calculate summary statistics"""
        strategy_counts = {option: 0 for option in REASONING_STRATEGIES.keys()}

        for item in metrics:
            for strategy in item['strategies_found']:
                strategy_counts[strategy] += 1

        total = len(metrics)
        summary = {
            'total_items': total,
            'strategies': {}
        }

        for option, count in strategy_counts.items():
            percentage = (count / total) * 100 if total > 0 else 0
            summary['strategies'][option] = {
                'name': REASONING_STRATEGIES[option],
                'count': count,
                'percentage': round(percentage, 2)
            }

        return summary

    def print_summary(self, summary: Dict, total: int):
        """Print summary statistics"""
        print("\n=== Strategy Usage Summary ===")
        for option, stats in summary['strategies'].items():
            print(f"{option}) {stats['name']}: {stats['count']}/{total} ({stats['percentage']:.1f}%)")


if __name__ == '__main__':
    import sys

    # Default paths
    results_path = 'data/MATH500/deepseek/p1/results.json'

    # Allow command line override
    if len(sys.argv) > 1:
        results_path = sys.argv[1]

    # Determine output path (same directory as results.json)
    output_path = str(Path(results_path).parent / 'metrics.json')

    analyzer = ReasoningAnalyzer()
    analyzer.analyze_results(results_path, output_path)
