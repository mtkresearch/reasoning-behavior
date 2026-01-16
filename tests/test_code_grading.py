"""
測試 run_experiment.py 中的代碼 grading 邏輯
"""
import pytest
import sys
import os

# Add parent directory to path to import run_experiment
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_experiment import create_code_grading_tasks, detect_dataset_type
from core import extract_code_blocks


class TestDetectDatasetType:
    """測試數據集類型檢測"""

    def test_detect_codeforces_dataset(self):
        """測試檢測 CodeForces 數據集"""
        results = [
            {'unique_id': 'codeforces-1971A-0'},
            {'unique_id': 'codeforces-1971B-1'}
        ]
        assert detect_dataset_type(results) == 'code'

    def test_detect_math_dataset(self):
        """測試檢測數學數據集"""
        results = [
            {'unique_id': 'aime2025-I-0-0'},
            {'unique_id': 'aime2025-I-1-1'}
        ]
        assert detect_dataset_type(results) == 'math'

    def test_empty_results(self):
        """測試空結果列表"""
        results = []
        assert detect_dataset_type(results) == 'math'  # 預設為 math


class TestCreateCodeGradingTasksNewFormat:
    """測試創建代碼 grading 任務（新的 baseline 格式）"""

    def test_create_tasks_with_new_format(self):
        """測試使用新的 baseline 格式創建任務"""
        results = [
            {
                'unique_id': 'codeforces-1971A-0',
                'question_id': 0,
                'generation_success': True,
                'result': {
                    'answer': '**Solution**\n...\n```cpp\n#include <iostream>\nint main(){return 0;}\n```'
                },
                'test_cases': [['input1', 'output1'], ['input2', 'output2']]
            }
        ]

        tasks = create_code_grading_tasks(results)

        assert len(tasks) == 1
        assert tasks[0]['type'] == 'code_execution'
        assert tasks[0]['unique_id'] == 'codeforces-1971A-0'
        assert tasks[0]['question_id'] == 0
        assert '#include <iostream>' in tasks[0]['code']
        assert len(tasks[0]['test_cases']) == 2
        assert tasks[0]['test_cases'][0] == ['input1', 'output1']

    def test_extract_code_from_result_answer(self):
        """測試從 result.answer 提取代碼"""
        results = [
            {
                'unique_id': 'codeforces-1971A-0',
                'question_id': 0,
                'generation_success': True,
                'result': {
                    'traj': 'Some reasoning...',
                    'answer': 'Explanation text\n\n```cpp\n#include <iostream>\nint main(){return 0;}\n```\n\nMore text'
                },
                'test_cases': [['in', 'out']]
            }
        ]

        tasks = create_code_grading_tasks(results)

        assert len(tasks) == 1
        assert '#include <iostream>' in tasks[0]['code']
        assert 'Explanation' not in tasks[0]['code']
        assert 'More text' not in tasks[0]['code']

    def test_missing_code_blocks(self):
        """測試沒有代碼塊的情況"""
        results = [
            {
                'unique_id': 'codeforces-1971A-0',
                'question_id': 0,
                'generation_success': True,
                'result': {
                    'answer': 'No code blocks in this answer'
                },
                'test_cases': [['in', 'out']]
            }
        ]

        tasks = create_code_grading_tasks(results)

        assert len(tasks) == 1
        assert tasks[0]['code'] == ""

    def test_empty_test_cases(self):
        """測試空的測試用例"""
        results = [
            {
                'unique_id': 'codeforces-1971A-0',
                'question_id': 0,
                'generation_success': True,
                'result': {
                    'answer': '```cpp\nint main(){}\n```'
                },
                'test_cases': []
            }
        ]

        tasks = create_code_grading_tasks(results)

        assert len(tasks) == 1
        assert tasks[0]['test_cases'] == []

    def test_multiple_code_blocks_prefer_cpp(self):
        """測試多個代碼塊時優先選擇 C++ 代碼"""
        results = [
            {
                'unique_id': 'codeforces-1971A-0',
                'question_id': 0,
                'generation_success': True,
                'result': {
                    'answer': '''Algorithm:
```
read n
print n
```

Implementation:
```cpp
#include <iostream>
int main() { int n; std::cin >> n; std::cout << n; }
```
'''
                },
                'test_cases': [['5', '5']]
            }
        ]

        tasks = create_code_grading_tasks(results)

        assert len(tasks) == 1
        assert '#include <iostream>' in tasks[0]['code']
        assert 'read n' not in tasks[0]['code']  # 不應包含偽代碼


class TestCodeExecution:
    """測試代碼執行和判定"""

    def test_execute_code_task_all_pass(self):
        """測試所有測試用例通過"""
        from core import compile_and_execute_cpp, normalize_output

        task = {
            'type': 'code_execution',
            'unique_id': 'codeforces-test-0',
            'question_id': 0,
            'code': '#include <iostream>\nint main() { std::cout << "Hello\\n"; return 0; }',
            'test_cases': [['', 'Hello']],
            'result': {'unique_id': 'codeforces-test-0'}
        }

        # Execute test cases
        all_passed = True
        execution_details = []

        for test_input, expected_output in task['test_cases']:
            exec_result = compile_and_execute_cpp(
                code=task['code'],
                test_input=test_input,
                timeout=2
            )

            # Check status and output
            if exec_result['status'] != 'AC':
                all_passed = False
            elif normalize_output(exec_result['output']) != normalize_output(expected_output):
                all_passed = False
                exec_result['status'] = 'WA'

            execution_details.append({
                'input': test_input[:50],
                'expected': expected_output[:50],
                'actual': exec_result['output'][:50],
                'status': exec_result['status']
            })

        assert all_passed is True
        assert execution_details[0]['status'] == 'AC'

    def test_execute_code_task_wrong_answer(self):
        """測試錯誤答案"""
        from core import compile_and_execute_cpp, normalize_output

        task = {
            'type': 'code_execution',
            'unique_id': 'codeforces-test-0',
            'question_id': 0,
            'code': '#include <iostream>\nint main() { std::cout << "World\\n"; return 0; }',
            'test_cases': [['', 'Hello']],
            'result': {'unique_id': 'codeforces-test-0'}
        }

        # Execute test cases
        all_passed = True

        for test_input, expected_output in task['test_cases']:
            exec_result = compile_and_execute_cpp(
                code=task['code'],
                test_input=test_input,
                timeout=2
            )

            if exec_result['status'] != 'AC':
                all_passed = False
            elif normalize_output(exec_result['output']) != normalize_output(expected_output):
                all_passed = False

        assert all_passed is False
