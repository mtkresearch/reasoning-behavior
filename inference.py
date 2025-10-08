import json
import random
from pathlib import Path

from tqdm import tqdm
from llm_client import LLMClient, Task, Request

MAX_WOKERS = 40
MAX_TRY = 5


def _get_sys_prompt(system_type):
    if system_type == 'p1':
        sys_prompt = "You are a helpful assistant"
    elif system_type == 'p2':
        sys_prompt = "You are a helpful assistant. In your thinking, prioritize using reverse thinking approaches whenever applicable."

    return sys_prompt

def save_result(data, out_dir):
    with open(out_dir + '/results.json', 'w') as fw:
        json.dump(data, fw, ensure_ascii=False, indent=2)

def save_result_incremental(item, index, out_dir):
    with open(out_dir + '/results.jsonl', 'a') as fw:
        fw.write(json.dumps({'index': index, **item}, ensure_ascii=False) + '\n')

def restore_from_jsonl(data, jsonl_path):
    """Restore results from JSONL file into data array"""
    if not Path(jsonl_path).exists():
        return data

    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                index = item.pop('index')
                data[index] = item
    return data


def _load_math500():
    """Load MATH500 dataset"""
    data = [json.loads(line.strip()) for line in open('/mnt/shared/p01/yc/datasets/MATH-500/test.jsonl').readlines()]
    return data, 'problem'


def _load_aime2025(target):
    """Load AIME2025 dataset"""
    # Parse repeat count from target (e.g., AIME2025__R5 means repeat 5 times)
    if '__R' in target:
        repeat_count = int(target.split('__R')[1])
    else:
        repeat_count = 1

    data = []
    # Process both AIME 2025 jsonl files
    for jsonl_file in ['aime2025-I.jsonl', 'aime2025-II.jsonl']:
        jsonl_path = f'/mnt/shared/p01/yc/datasets/AIME2025/{jsonl_file}'
        filename = jsonl_file.replace('.jsonl', '')

        with open(jsonl_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        for row_idx, line in enumerate(lines):
            item = json.loads(line)
            # Repeat each problem repeat_count times
            for repeat_idx in range(repeat_count):
                unique_id = f"{filename}-{row_idx}-{repeat_idx}"
                data.append({
                    'unique_id': unique_id,
                    'question': item['question'],
                    'answer': item['answer'],
                })

    return data, 'question'


def inference(target, policy_model_type, system_type, out_dir, can_restore=False):

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    client = LLMClient()

    path_results_jsonl = out_dir + '/results.jsonl'

    # Load dataset based on target
    if target == 'MATH500':
        data, col_problem = _load_math500()
    elif target.startswith('AIME2025'):
        data, col_problem = _load_aime2025(target)
    else:
        raise ValueError(f"Unknown target: {target}")

    # Restore results if needed
    if can_restore and Path(path_results_jsonl).exists():
        data = restore_from_jsonl(data, path_results_jsonl)

    index_query_pairs = [(i, x[col_problem]) for i, x in enumerate(data) if 'result' not in x]

    # Prepare tasks
    sys_prompt = _get_sys_prompt(system_type)
    tasks = [Task(index=i, request=Request(query=query, model_type=policy_model_type, system_prompt=sys_prompt, reasoning_on=True))
             for i, query in index_query_pairs]

    # Shuffle tasks
    random.shuffle(tasks)

    # Process with retries
    for attempt in range(MAX_TRY):
        failed_tasks = []

        for task in tqdm(client.generate_concurrent(tasks, max_workers=MAX_WOKERS), total=len(tasks), desc=f"Attempt {attempt+1}"):
            if not task.response.success:
                print(f"Error index {task.index}: Generation failed - {task.response.err_message}")
                failed_tasks.append(task)
                continue

            try:
                data[task.index]['result'] = {
                    'traj': task.response.reasoning_content,
                    'answer': task.response.content,
                    'sys_prompt': sys_prompt,
                    'elapsed_seconds': task.response.elapsed_seconds
                }
                # Save incrementally to JSONL
                save_result_incremental(data[task.index], task.index, out_dir)
            except Exception as e:
                print(f"Error index {task.index}: {e}")
                failed_tasks.append(task)

        if not failed_tasks:
            break
        tasks = failed_tasks

    # Final save to JSON format
    save_result(data, out_dir)


if __name__ == '__main__':
    import sys

    # Default parameters
    target = 'AIME2025__R10'
    policy_model_type = 'deepseek'
    system_type = 'p1'

    # Allow command line override
    assert len(sys.argv) == 1 + 0 or len(sys.argv) == 1 + 3
    if len(sys.argv) > 1:
        target = sys.argv[1]
    if len(sys.argv) > 2:
        policy_model_type = sys.argv[2]
    if len(sys.argv) > 3:
        system_type = sys.argv[3]

    out_dir = f'data/{target}/{policy_model_type}/{system_type}/'
    inference(target, policy_model_type, system_type, out_dir, can_restore=True)
