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


class ConsistencyInference:
    def __init__(self, model_type: str = 'deepseek'):
        self.client = LLMClient()
        self.model_type = model_type

    def analyze_single_item(self, item: Dict, index: int) -> Dict:
        """Analyze a single problem's reasoning trajectory using concurrent strategy checking"""
        traj = item['traj']

        # Create tasks for all strategies
        tasks = []
        for option, strategy_name in REASONING_STRATEGIES.items():
            task = Task(
                index=ord(option) - ord('A'),  # Convert A-I to 0-8
                request=Request(
                    query=DIRECT_REASONING_WAY_SELECTION.format(traj=traj, option=option),
                    model_type=self.model_type,
                    system_prompt="You are a helpful assistant",
                    extra_body={"chat_template_kwargs": {"thinking": False}}
                ),
                metadata={'option': option, 'strategy_name': strategy_name}
            )
            tasks.append(task)

        method_types = []
        cot = {}

        # Process all strategies concurrently
        for completed_task in self.client.generate_concurrent(tasks, max_workers=MAX_WORKERS):
            option = completed_task.metadata['option']
            strategy_name = completed_task.metadata['strategy_name']

            if not completed_task.response.success:
                raise Exception(f"Generation failed for strategy {option} in item {index}")

            try:
                has_strategy = '\\boxed{YES}' in completed_task.response.content
                cot[option] = {
                    'name': strategy_name,
                    'response': completed_task.response.content,
                    'found': has_strategy
                }
                if has_strategy:
                    method_types.append(option)
            except Exception as e:
                print(f"Error processing strategy {option} for item {index}: {e}")
                cot[option] = {
                    'name': strategy_name,
                    'error': str(e),
                    'found': False
                }

        return {
            'method_types': sorted(method_types),
            'cot': cot
        }

    def process_consistency_data(self, input_path: str, output_path: str):
        """Process consistency_data.json and save results for specific model_type

        Output format: [{"unique_id": "...", "method_types": [...], "cot": {...}}, ...]
        """
        # Load input data
        with open(input_path, 'r') as f:
            data = json.load(f)

        print(f"Processing {len(data)} items with model_type={self.model_type}...")

        results = []
        for i, item in enumerate(tqdm(data, desc=f"Analyzing with {self.model_type}")):
            analysis = self.analyze_single_item(item, i)

            # Build result item - unique_id, method_types and cot
            result_item = {
                'unique_id': item.get('unique_id', ''),
                'method_types': analysis['method_types'],
                'cot': analysis['cot']
            }
            results.append(result_item)

            # Save incrementally
            with open(output_path, 'w') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Processing complete. Results saved to {output_path}")


if __name__ == '__main__':
    import sys

    # Default paths
    input_path = 'consistency_data/consistency_data.json'

    # Determine model_type from command line (default to 'deepseek')
    model_type = sys.argv[1] if len(sys.argv) > 1 else 'deepseek'

    # Override input path if provided
    if len(sys.argv) > 2:
        input_path = sys.argv[2]

    # Output path based on model_type
    output_path = str(Path(input_path).parent / f'consistency_{model_type}.json')

    # Create analyzer
    analyzer = ConsistencyInference(model_type=model_type)

    # Process the data
    analyzer.process_consistency_data(input_path, output_path)
