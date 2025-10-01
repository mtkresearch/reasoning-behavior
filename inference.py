import json
from pathlib import Path

from tqdm import tqdm
from openai import OpenAI


def _get_sys_prompt(model_type, system_type):
    if model_type == 'deepseek':
        if system_type == 'p1':
            sys_prompt = "You are a helpful assistant"
        elif system_type == 'p2':
            sys_prompt = "You are a helpful assistant. In your thinking, prioritize using reverse thinking approaches whenever applicable."

    return sys_prompt

def generate(model_type, system_type, query):
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8001/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id


    sys_prompt = _get_sys_prompt(model_type, system_type)

    if model_type == 'deepseek':
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "<think>Hmm</think>I am DeepSeek"},
            {"role": "user", "content": query},
        ]
        extra_body = {"chat_template_kwargs": {"thinking": True}}

    response = client.chat.completions.create(
        model=model, messages=messages, extra_body=extra_body
    )
    content = response.choices[0].message.content
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
    MAX_TRY = 5
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if target == 'MATH500':
        path_results = out_dir + '/results.json'
        if can_restore and Path(path_results).exists():
            data = json.load(open(path_results))
        else:
            data = [json.loads(line.strip()) for line in open('/mnt/shared/p01/yc/datasets/MATH-500/test.jsonl').readlines()]

        col_problem = 'problem'

    index_query_pairs = [(i, x[col_problem]) for i, x in enumerate(data) if 'result' not in x]

    for i, query in tqdm(index_query_pairs):
        for _ in range(MAX_TRY):
            try:
                response = generate(model_type, system_type, query)
                traj, answer = parse_response(model_type, response)
                data[i]['result'] = {
                    'traj': traj,
                    'answer': answer,
                    'sys_prompt': _get_sys_prompt(model_type, system_type)
                }
                break
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                print(e)
        save_result(data, out_dir)
    # save_result(data, out_dir)


if __name__ == '__main__':
    target = 'MATH500'
    model_type = 'deepseek'
    system_type = 'p1'
    out_dir = f'data/{target}/{model_type}/{system_type}/'
    inference(target, model_type, system_type, out_dir, can_restore=True)
