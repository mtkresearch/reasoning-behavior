import json
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm
from behavior_extract import ReasoningAnalyzer


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
    output_path = str(Path(input_path).parent / f'consistency3_{judge_model_type}.json')

    # Handle reasoning_on flag
    reasoning_on = False
    if judge_model_type == 'gpt-oss-reasoning':
        reasoning_on = True
        judge_model_type = 'gpt-oss'

    # Create analyzer
    analyzer = ReasoningAnalyzer(judge_model_type=judge_model_type, reasoning_on=reasoning_on)
    analyzer.analyze_results(
        input_path, 
        output_path, 
        parsing_func=lambda item: (item['question'], item['traj'], item['unique_id']),
        limit=None
    )
