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

MAX_WORKERS = 50
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
    def __init__(self, judge_model_type: str = 'deepseek'):
        self.client = LLMClient()
        self.judge_model_type = judge_model_type

    def analyze_results(self, results_path: str, output_path: str):
        """Analyze all items in results.json and save to metrics.json"""
        # Load results
        with open(results_path, 'r') as f:
            data = json.load(f)

        # Filter items that have results
        items_to_analyze = [(i, item) for i, item in enumerate(data) if 'result' in item]

        print(f"Analyzing {len(items_to_analyze)} items...")

        # Create all tasks (items × 9 strategies)
        all_tasks = []
        for i, item in items_to_analyze:
            traj = item['result']['traj']
            for option, strategy_name in REASONING_STRATEGIES.items():
                task = Task(
                    index=len(all_tasks),
                    request=Request(
                        query=DIRECT_REASONING_WAY_SELECTION.format(traj=traj, option=option),
                        model_type=self.judge_model_type,
                        system_prompt="You are a helpful assistant",
                        reasoning_on=False
                    ),
                    metadata={
                        'item_index': i,
                        'unique_id': item.get('unique_id', ''),
                        'option': option,
                        'strategy_name': strategy_name
                    }
                )
                all_tasks.append(task)

        print(f"Total tasks: {len(all_tasks)} ({len(items_to_analyze)} items × 9 strategies)")

        # Collect results, group by item_index
        results_by_item = {}
        failed_items = set()

        for completed_task in tqdm(
            self.client.generate_concurrent(all_tasks, max_workers=MAX_WORKERS),
            total=len(all_tasks),
            desc="Processing all strategies"
        ):
            item_idx = completed_task.metadata['item_index']
            option = completed_task.metadata['option']
            strategy_name = completed_task.metadata['strategy_name']

            if item_idx not in results_by_item:
                results_by_item[item_idx] = {
                    'unique_id': completed_task.metadata['unique_id'],
                    'cot_collection': {},
                    'strategies_found': []
                }

            # Check if failed
            if not completed_task.response.success:
                print(f"\nTask failed for item {item_idx}, strategy {option}: {completed_task.response.err_message}")
                failed_items.add(item_idx)
                continue

            # Parse result
            try:
                has_strategy = '\\boxed{YES}' in completed_task.response.content
                results_by_item[item_idx]['cot_collection'][option] = {
                    'name': strategy_name,
                    'response': completed_task.response.content,
                    'found': has_strategy
                }
                if has_strategy:
                    results_by_item[item_idx]['strategies_found'].append(option)
            except Exception as e:
                print(f"\nError processing item {item_idx}, strategy {option}: {e}")
                failed_items.add(item_idx)

        # Filter complete results (all 9 strategies succeeded)
        metrics = []
        for item_idx in sorted(results_by_item.keys()):
            if item_idx in failed_items:
                print(f"Skipping item {item_idx} due to failures")
                continue

            if len(results_by_item[item_idx]['cot_collection']) != 9:
                print(f"Skipping item {item_idx}: only {len(results_by_item[item_idx]['cot_collection'])}/9 strategies completed")
                continue

            metrics.append({
                'index': item_idx,
                'unique_id': results_by_item[item_idx]['unique_id'],
                'strategies_found': sorted(results_by_item[item_idx]['strategies_found']),
                'cot_collection': results_by_item[item_idx]['cot_collection']
            })

        # Save complete results
        summary = self.calculate_summary(metrics)
        output_data = {
            'summary': summary,
            'metrics': metrics
        }
        with open(output_path, 'w') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\nAnalysis complete. Metrics saved to {output_path}")
        print(f"Completed: {len(metrics)}/{len(items_to_analyze)} items")
        print(f"Failed/Incomplete: {len(items_to_analyze) - len(metrics)} items")

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
    judge_model_type = 'deepseek'

    # Allow command line override
    assert len(sys.argv) == 1 + 0 or len(sys.argv) == 1 + 1 or len(sys.argv) == 1 + 2
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    if len(sys.argv) > 2:
        judge_model_type = sys.argv[2]

    # Determine output path (same directory as results.json)
    output_path = str(Path(results_path).parent / 'metrics.json')

    analyzer = ReasoningAnalyzer(judge_model_type=judge_model_type)
    analyzer.analyze_results(results_path, output_path)
