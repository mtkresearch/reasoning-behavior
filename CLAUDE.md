# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Reasoning Behavior Analysis** research project for studying and evaluating Large Language Models' (LLMs) reasoning capabilities and behavior patterns. The project focuses on:

1. **Reasoning Strategy Analysis**: Identifying and classifying thinking strategies (reflection, problem decomposition, hypothesis testing, etc.)
2. **Reasoning Robustness Testing**: Testing reasoning reliability through shuffling, truncation, and noise insertion
3. **Model Performance Comparison**: Comparing different models and reasoning approaches on mathematical problems

### Directory Structure Overview

```
reasoning-behavior/
├── behavior_analysis/           # Deprecated
│
├── model_serving/              # VLLM server management
│   └── vllm-multi-node/        # Multi-node deployment
│
├── design/                     # Design documentation
│   └── knowledge/              # Knowledge base
│
├── datasets/                   # Dataset sources
│   └── AIME2025/
│
├── data/                       # Source experimental results
│   └── [dataset]__[R{n}]/     # Baseline results with reasoning
│
├── exp/                        # Pipeline experiment results
│   └── [processor1]/[processor2]/.../results.json
│
├── llm_client.py              # Core LLM abstraction
├── core.py                    # Shared utilities for all experiments
├── pipeline.py                # Pipeline framework for processor composition
├── run_experiment.py          # Pipeline-based experiment runner
└── view_experiment.py         # Web visualization server
```

## Core Architecture

### Project Structure

The codebase is organized into two main layers:

**Root Directory** - Core infrastructure and experiment scripts:
- `llm_client.py` - Central LLM client abstraction
- `core.py` - Shared utilities (data loading, parsing, text processing, prompt templates)
- Experiment scripts: `comparison_shuffle_reasoning.py`, `insert_noise_reasoning.py`, etc.
- Analysis utilities: `analyze_shuffle_results.py`, `filter_by_consensus.py`, etc.

### LLM Client Layer (`llm_client.py`)

The central abstraction for all LLM interactions:
- Supports both OpenRouter (remote) and local VLLM servers
- Provides synchronous and concurrent generation interfaces
- Supports multiple models: DeepSeek, GPT-OSS, Qwen3
- Key classes: `Request`, `Response`, `Task`, `LLMClient`

### Core Utilities Layer (`core.py`)

Shared utilities used across all experiment scripts to eliminate code duplication:

**Debug Utilities:**
- `DEBUG` - Environment variable flag for debug mode
- `debug_print(msg)` - Conditional debug message printing

**Data Loading:**
- `load_existing_results(path)` - Load existing results.json files
- `load_existing_grades(path)` - Load existing grades.json with index→correctness mapping

**Parsing Functions:**
- `parse_answer_from_completion(text)` - Extract final answer from model completion
- `parse_yes_no_response(text)` - Parse YES/NO responses from grading tasks

**Text Processing:**
- `clean_multiple_newlines(text)` - Replace multiple consecutive newlines with single newline
- `extract_nonempty_lines(text)` - Extract non-empty lines from text

**Prompt Construction:**
- `build_gpt_oss_prompt_with_reasoning(question, reasoning, ...)` - Build GPT-OSS completion prompts with prefilled reasoning

**Prompt Templates:**
- `GRADING_PROMPT` - Standard mathematical answer grading template
- `SAME_ANSWER_PROMPT` - Template for comparing answer equivalence

**Usage Example:**
```python
from core import (
    debug_print,
    load_existing_results,
    parse_answer_from_completion,
    build_gpt_oss_prompt_with_reasoning,
    GRADING_PROMPT
)
```

### Main Components

**Pipeline Experiment Framework**:
- `run_experiment.py` - Configurable pipeline for processing reasoning (mask, truncate, shuffle, insert) with automatic result caching and grading
- `view_experiment.py` - Web visualization server for browsing experiment results with tree structure and conditional probability analysis
- `pipeline.py` - Core pipeline infrastructure with processor composition
- Output structure: `exp/<processor1>/<processor2>/.../results.json`

## Datasets

- **MATH500**: 500 mathematical problems
- **AIME2025**: American Invitational Mathematics Examination problems
  - Supports repeated sampling with suffix (e.g., `AIME2025__R10` for 10 repetitions)

## Common Commands

### Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Experiment Scripts

```bash
# Run processing pipeline (mask, truncate, shuffle, insert) and evaluate
python run_experiment.py --flow "mask('number'),shuffle('line')" \
    --results_path data/AIME2025__R10/gpt-oss/p1/results.json \
    --model_type gpt-oss

# Auto-generate output path based on flow
python run_experiment.py --flow "truncate('last_ratio',ratio=0.3),mask('alphabet')"

# View experiment results in web interface
python view_experiment.py [--port 5000] [--host 127.0.0.1] [--exp-dir exp/]
```

## Key Design Patterns

1. **Concurrent Execution**: Uses `ThreadPoolExecutor` for large-scale parallel inference with configurable `CONCURRENCY` settings
2. **Incremental Saving**: JSONL format with automatic resume capability (checks existing results before re-running)
3. **Retry Mechanism**: `MAX_TRY` parameter for handling API failures
4. **Data Classes**: Uses `@dataclass` for clear data structures (Request, Response, Task)
5. **Modular Experiments**: Each experiment type has its own script sharing the core infrastructure (`llm_client.py` and `core.py`)
6. **DRY Principle**: Common utilities extracted to `core.py` to eliminate code duplication across experiment scripts

## Important Configuration

### Environment Variables
Create a `.env` file with:
```bash
OPENROUTER_API_KEY=your_key_here
# Optional: OPENROUTER_SITE_URL, OPENROUTER_SITE_NAME for rankings
```

Reference implementation can be found in `design/knowledge/openrouter.md`.

### Model Endpoints
- **OpenRouter**: Configured via `OPENROUTER_API_KEY`
- **Local VLLM**: Default ports defined in `llm_client.py`

### Concurrency Settings
Most scripts have `CONCURRENCY` parameter (typically 4-50) controlling parallel execution. Adjust based on:
- API rate limits (OpenRouter)
- Server capacity (VLLM)
- Memory constraints

## Data Organization

```
data/                                # Source experimental results
└── [dataset]__[R{repetitions}]/     # e.g., AIME2025__R10
    └── [model]/                     # e.g., gpt-oss, deepseek, qwen3
        └── [prompt]/                # e.g., p1, p2, p3
            └── results.json         # Baseline reasoning results

exp/                                 # Pipeline experiment results
└── [processor1]/                    # e.g., mask_number
    └── [processor2]/                # e.g., shuffle_line
        └── .../                     # nested processors
            ├── results.json         # Final results with metadata
            ├── results_stage1.jsonl # Generation stage (cached)
            └── results_stage2.jsonl # Grading stage (cached)

datasets/                            # Dataset sources
└── AIME2025/                        # AIME 2025 problems
```

## Working with This Codebase

### Code Organization Principles

1. **Pipeline Architecture**: Use `run_experiment.py` with `--flow` parameter to compose processing steps (mask, truncate, shuffle, insert)

2. **Shared utilities layer**: `core.py` provides common functions:
   - Data loading and parsing
   - Text processing utilities
   - Prompt construction helpers
   - Standard prompt templates

3. **Results are cached**: JSONL-based incremental saving with automatic resume capability
   - Stage 1: Generation results (`results_stage1.jsonl`)
   - Stage 2: Grading results (`results_stage2.jsonl`)
   - Final: Aggregated JSON (`results.json`)

4. **Visualization**: Use `view_experiment.py` to browse results with tree structure and conditional probability analysis
