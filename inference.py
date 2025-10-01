import json
from pathlib import Path

from tqdm import tqdm
from llm_client import LLMClient, Task

MAX_WOKERS = 10
MAX_TRY = 3


def _get_sys_prompt(model_type, system_type):
    if model_type == 'deepseek':
        if system_type == 'p1':
            sys_prompt = "You are a helpful assistant"
        elif system_type == 'p2':
            sys_prompt = "You are a helpful assistant. In your thinking, prioritize using reverse thinking approaches whenever applicable."

    return sys_prompt

def parse_response(model_type, response):
    if model_type == 'deepseek':
        part1, part2 = response.split('</think>')
        traj = part1.split('<think>')[-1]
        answer = part2.split('</ans>')[0]
    
    return traj, answer


def save_result(data, out_dir):
    with open(out_dir + '/results.json', 'w') as fw:
        json.dump(data, fw, ensure_ascii=False, indent=2)


def inference(target, model_type, system_type, out_dir, can_restore=False):

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    client = LLMClient()

    if target == 'MATH500':
        path_results = out_dir + '/results.json'
        if can_restore and Path(path_results).exists():
            data = json.load(open(path_results))
        else:
            data = [json.loads(line.strip()) for line in open('/mnt/shared/p01/yc/datasets/MATH-500/test.jsonl').readlines()]

        col_problem = 'problem'

    elif target.startswith('AIME2025'):
        # Parse repeat count from target (e.g., AIME2025__R5 means repeat 5 times)
        if '__R' in target:
            repeat_count = int(target.split('__R')[1])
        else:
            repeat_count = 1

        path_results = out_dir + '/results.json'
        if can_restore and Path(path_results).exists():
            data = json.load(open(path_results))
        else:
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

        col_problem = 'question'

    index_query_pairs = [(i, x[col_problem]) for i, x in enumerate(data) if 'result' not in x]

    # Prepare tasks
    sys_prompt = _get_sys_prompt(model_type, system_type)
    tasks = [Task(index=i, query=query, model_type=model_type, system_prompt=sys_prompt)
             for i, query in index_query_pairs]

    # Process with retries
    for attempt in range(MAX_TRY):
        failed_tasks = []

        for response in tqdm(client.generate_concurrent(tasks, max_workers=MAX_WOKERS), total=len(tasks), desc=f"Attempt {attempt+1}"):
            try:
                traj, answer = parse_response(model_type, response.content)
                data[response.index]['result'] = {
                    'traj': traj,
                    'answer': answer,
                    'sys_prompt': sys_prompt
                }
            except Exception as e:
                print(f"Error index {response.index}: {e}")
                failed_tasks.append([t for t in tasks if t.index == response.index][0])

            save_result(data, out_dir)

        if not failed_tasks:
            break
        tasks = failed_tasks


if __name__ == '__main__':
    import sys

    # Default parameters
    target = 'AIME2025__R10'
    model_type = 'deepseek'
    system_type = 'p2'

    # Allow command line override
    if len(sys.argv) > 1:
        target = sys.argv[1]
    if len(sys.argv) > 2:
        model_type = sys.argv[2]
    if len(sys.argv) > 3:
        system_type = sys.argv[3]

    out_dir = f'data/{target}/{model_type}/{system_type}/'
    inference(target, model_type, system_type, out_dir, can_restore=True)
