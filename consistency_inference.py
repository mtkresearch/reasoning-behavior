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

MAX_WORKERS = 100
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


class ConsistencyInference:
    def __init__(self, judge_model_type: str = 'deepseek'):
        self.client = LLMClient()
        self.judge_model_type = judge_model_type

    def _process_single_task(self, task: Task) -> Dict:
        """Process a single task and extract strategy information"""
        option = task.metadata['option']
        strategy_name = task.metadata['strategy_name']

        if not task.response.success:
            return {
                'option': option,
                'name': strategy_name,
                'error': task.response.err_message,
                'found': False
            }

        try:
            has_strategy = '\\boxed{YES}' in task.response.content
            return {
                'option': option,
                'name': strategy_name,
                'response': task.response.content,
                'found': has_strategy
            }
        except Exception as e:
            return {
                'option': option,
                'name': strategy_name,
                'error': str(e),
                'found': False
            }

    def process_consistency_data(self, input_path: str, output_path: str):
        """Process consistency_data.json and save results for specific model_type

        Output format: [{"unique_id": "...", "method_types": [...], "cot": {...}}, ...]
        """
        # Load input data
        with open(input_path, 'r') as f:
            data = json.load(f)

        print(f"Processing {len(data)} items with judge_model_type={self.judge_model_type}...")

        # Create all tasks at once (items × strategies)
        all_tasks = []
        for item_idx, item in enumerate(data):
            traj = item['traj']
            for option, strategy_name in REASONING_STRATEGIES.items():
                task = Task(
                    index=item_idx * len(REASONING_STRATEGIES) + (ord(option) - ord('A')),
                    request=Request(
                        query=DIRECT_REASONING_WAY_SELECTION.format(traj=traj, option=option),
                        model_type=self.judge_model_type,
                        system_prompt="You are a helpful assistant",
                        reasoning_on=False
                    ),
                    metadata={
                        'item_idx': item_idx,
                        'option': option,
                        'strategy_name': strategy_name,
                        'unique_id': item.get('unique_id', '')
                    }
                )
                all_tasks.append(task)

        # Initialize results structure
        results = [{
            'unique_id': item.get('unique_id', ''),
            'method_types': [],
            'cot': {}
        } for item in data]

        # Process all tasks concurrently with retries
        for attempt in range(MAX_TRY):
            failed_tasks = []

            for task in tqdm(self.client.generate_concurrent(all_tasks, max_workers=MAX_WORKERS),
                           total=len(all_tasks),
                           desc=f"Attempt {attempt+1}/{MAX_TRY}"):
                if not task.response.success:
                    print(f"Error task {task.index}: Generation failed - {task.response.err_message}")
                    failed_tasks.append(task)
                    continue

                try:
                    item_idx = task.metadata['item_idx']
                    result_info = self._process_single_task(task)
                    option = result_info['option']

                    # Store the strategy analysis result
                    results[item_idx]['cot'][option] = {
                        k: v for k, v in result_info.items() if k != 'option'
                    }

                    # Add to method_types if found
                    if result_info['found']:
                        results[item_idx]['method_types'].append(option)

                except Exception as e:
                    print(f"Error processing task {task.index}: {e}")
                    failed_tasks.append(task)

            if not failed_tasks:
                break
            all_tasks = failed_tasks

        # Sort method_types for each result
        for result in results:
            result['method_types'] = sorted(result['method_types'])

        # Save final results
        with open(output_path, 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Processing complete. Results saved to {output_path}")


if __name__ == '__main__':
    import sys

    # Default paths
    judge_model_type = 'deepseek'
    input_path = 'consistency_data/consistency_data.json'

    # Allow command line override
    if len(sys.argv) > 1:
        judge_model_type = sys.argv[1]
    if len(sys.argv) > 2:
        input_path = sys.argv[2]

    # Output path based on judge_model_type
    output_path = str(Path(input_path).parent / f'consistency_{judge_model_type}.json')

    # Create analyzer
    analyzer = ConsistencyInference(judge_model_type=judge_model_type)

    # Process the data
    analyzer.process_consistency_data(input_path, output_path)
