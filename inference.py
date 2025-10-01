import json
from pathlib import Path

from tqdm import tqdm
from llm_client import LLMClient, Task

MAX_WOKERS = 5
MAX_TRY = 5


def _get_sys_prompt(model_type, system_type):
    if model_type == 'deepseek':
        if system_type == 'p1':
            sys_prompt = "You are a helpful assistant"
        elif system_type == 'p2':
            sys_prompt = "You are a helpful assistant. In your thinking, prioritize using reverse thinking approaches whenever applicable."

    return sys_prompt

def generate(model_type, system_type, query, client):
    sys_prompt = _get_sys_prompt(model_type, system_type)

    if model_type == 'deepseek':
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "<think>Hmm</think>I am DeepSeek"},
            {"role": "user", "content": query},
        ]
        extra_body = {"chat_template_kwargs": {"thinking": True}}

    content = client.generate(messages, extra_body)
    return content


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

    index_query_pairs = [(i, x[col_problem]) for i, x in enumerate(data) if 'result' not in x]

    # Prepare tasks
    sys_prompt = _get_sys_prompt(model_type, system_type)
    tasks = []
    for i, query in index_query_pairs:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "<think>Hmm</think>I am DeepSeek"},
            {"role": "user", "content": query},
        ]
        extra_body = {"chat_template_kwargs": {"thinking": True}}
        tasks.append(Task(index=i, messages=messages, extra_body=extra_body))

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
    target = 'MATH500'
    model_type = 'deepseek'
    system_type = 'p1'
    out_dir = f'data/{target}/{model_type}/{system_type}/'
    inference(target, model_type, system_type, out_dir, can_restore=True)
