#!/bin/bash

# Shuffle Reasoning Comparison Experiments (VLLM version)
# Compares normal vs shuffled reasoning with different truncation levels

# Configuration
RESULTS_PATH="data/AIME2025__R10/gpt-oss/p1/results.json"
GRADES_PATH="data/AIME2025__R10/gpt-oss/p1/grades.json"
JUDGE_MODEL="gpt-oss"
MAX_WORKERS=50


python comparison_shuffle_reasoning.py \
    --results_path ${RESULTS_PATH} \
    --grades_path ${GRADES_PATH} \
    --output_dir ./data/shuffle_comparison_exp/empty_question_f00/ \
    --del_last_line 0.0 \
    --seed 116 \
    --judge_model_type ${JUDGE_MODEL} \
    --max_workers ${MAX_WORKERS} \
    --empty_question

# Run experiments with different seeds and truncation levels
# Note: All experiments use the same results.json and grades.json
# The differences come from different seeds (for shuffling) and truncation levels

# echo "=================================="
# echo "Starting Shuffle Reasoning Experiments"
# echo "=================================="

# # Seed 116, Experiment 1
# python comparison_shuffle_reasoning.py \
#     --results_path ${RESULTS_PATH} \
#     --grades_path ${GRADES_PATH} \
#     --output_dir ./data/shuffle_comparison_exp/turncate_f05/ \
#     --del_last_line 0.5 \
#     --seed 116 \
#     --judge_model_type ${JUDGE_MODEL} \
#     --max_workers ${MAX_WORKERS}

# python comparison_shuffle_reasoning.py \
#     --results_path ${RESULTS_PATH} \
#     --grades_path ${GRADES_PATH} \
#     --output_dir ./data/shuffle_comparison_exp/turncate_f03/ \
#     --del_last_line 0.3 \
#     --seed 116 \
#     --judge_model_type ${JUDGE_MODEL} \
#     --max_workers ${MAX_WORKERS}

# python comparison_shuffle_reasoning.py \
#     --results_path ${RESULTS_PATH} \
#     --grades_path ${GRADES_PATH} \
#     --output_dir ./data/shuffle_comparison_exp/turncate_f01/ \
#     --del_last_line 0.1 \
#     --seed 116 \
#     --judge_model_type ${JUDGE_MODEL} \
#     --max_workers ${MAX_WORKERS}

# python comparison_shuffle_reasoning.py \
#     --results_path ${RESULTS_PATH} \
#     --grades_path ${GRADES_PATH} \
#     --output_dir ./data/shuffle_comparison_exp/turncate_f07/ \
#     --del_last_line 0.7 \
#     --seed 116 \
#     --judge_model_type ${JUDGE_MODEL} \
#     --max_workers ${MAX_WORKERS}

# python comparison_shuffle_reasoning.py \
#     --results_path ${RESULTS_PATH} \
#     --grades_path ${GRADES_PATH} \
#     --output_dir ./data/shuffle_comparison_exp/turncate_f09/ \
#     --del_last_line 0.9 \
#     --seed 116 \
#     --judge_model_type ${JUDGE_MODEL} \
#     --max_workers ${MAX_WORKERS}
