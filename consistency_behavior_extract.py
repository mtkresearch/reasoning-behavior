import json
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm
from behavior_extract import ReasoningAnalyzer, REASONING_STRATEGIES, DIRECT_REASONING_WAY_SELECTION, MAX_WORKERS


def process_consistency_data(input_path: str, output_path: str, judge_model_type: str = 'deepseek'):
    """Process consistency_data.json and save results for specific model_type

    Output format: [{"unique_id": "...", "method_types": [...], "cot": {...}}, ...]
    """
    from llm_client import Task, Request

    # Handle reasoning_on flag
    reasoning_on = False
    if judge_model_type == 'gpt-oss-reasoning':
        reasoning_on = True
        judge_model_type = 'gpt-oss'

    # Create analyzer
    analyzer = ReasoningAnalyzer(judge_model_type=judge_model_type, reasoning_on=reasoning_on)

    # Load input data
    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"Processing {len(data)} items with judge_model_type={judge_model_type}...")

    # Create all tasks at once (items × strategies)
    all_tasks = []
    for item_idx, item in enumerate(data):
        traj = item['traj']
        for option, strategy_name in REASONING_STRATEGIES.items():
            task = Task(
                index=len(all_tasks),
                request=Request(
                    query=DIRECT_REASONING_WAY_SELECTION.format(traj=traj, option=option),
                    model_type=judge_model_type,
                    system_prompt="You are a helpful assistant",
                    reasoning_on=reasoning_on
                ),
                metadata={
                    'item_index': item_idx,
                    'unique_id': item.get('unique_id', ''),
                    'option': option,
                    'strategy_name': strategy_name
                }
            )
            all_tasks.append(task)

    print(f"Total tasks: {len(all_tasks)} ({len(data)} items × 9 strategies)")

    # Collect results, group by item_index
    results_by_item = {}
    failed_items = set()

    for completed_task in tqdm(
        analyzer.client.generate_concurrent(all_tasks, max_workers=MAX_WORKERS),
        total=len(all_tasks),
        desc="Processing all strategies"
    ):
        item_idx = completed_task.metadata['item_index']
        option = completed_task.metadata['option']
        strategy_name = completed_task.metadata['strategy_name']

        if item_idx not in results_by_item:
            results_by_item[item_idx] = {
                'unique_id': completed_task.metadata['unique_id'],
                'method_types': [],
                'cot': {}
            }

        # Check if failed
        if not completed_task.response.success:
            print(f"\nTask failed for item {item_idx}, strategy {option}: {completed_task.response.err_message}")
            failed_items.add(item_idx)
            continue

        # Parse result
        try:
            has_strategy = '\\boxed{YES}' in completed_task.response.content
            results_by_item[item_idx]['cot'][option] = {
                'name': strategy_name,
                'response': completed_task.response.content,
                'found': has_strategy
            }
            if has_strategy:
                results_by_item[item_idx]['method_types'].append(option)
        except Exception as e:
            print(f"\nError processing item {item_idx}, strategy {option}: {e}")
            failed_items.add(item_idx)

    # Build final results in original order
    results = []
    for item_idx in range(len(data)):
        if item_idx in failed_items:
            print(f"Skipping item {item_idx} due to failures")
            # Still include the item with partial data
            if item_idx in results_by_item:
                result = results_by_item[item_idx]
                result['method_types'] = sorted(result['method_types'])
                results.append(result)
            else:
                results.append({
                    'unique_id': data[item_idx].get('unique_id', ''),
                    'method_types': [],
                    'cot': {}
                })
            continue

        if item_idx not in results_by_item:
            print(f"Missing item {item_idx}")
            results.append({
                'unique_id': data[item_idx].get('unique_id', ''),
                'method_types': [],
                'cot': {}
            })
            continue

        if len(results_by_item[item_idx]['cot']) != 9:
            print(f"Item {item_idx}: only {len(results_by_item[item_idx]['cot'])}/9 strategies completed")

        result = results_by_item[item_idx]
        result['method_types'] = sorted(result['method_types'])
        results.append(result)

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nProcessing complete. Results saved to {output_path}")
    print(f"Total items: {len(results)}")
    complete_count = sum(1 for r in results if len(r['cot']) == 9)
    print(f"Complete items (9/9 strategies): {complete_count}/{len(results)}")


if __name__ == '__main__':
    import sys

    # Default paths
    judge_model_type = 'gpt-oss-reasoning'
    input_path = 'consistency_data/consistency_data.json'

    # Allow command line override
    if len(sys.argv) > 1:
        judge_model_type = sys.argv[1]
    if len(sys.argv) > 2:
        input_path = sys.argv[2]

    # Output path based on judge_model_type
    output_path = str(Path(input_path).parent / f'consistency_{judge_model_type}.json')

    # Process the data
    process_consistency_data(input_path, output_path, judge_model_type)
