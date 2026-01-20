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
├── experiments/                # Experiment results root
│   ├── exp/                    # Pipeline experiment results
│   │   └── [processor1]/[processor2]/.../results.json
│   └── exp_*/                  # Dataset-specific results
│
├── llm_client.py              # Core LLM abstraction
├── core.py                    # Shared utilities for all experiments
├── pipeline.py                # Pipeline framework for processor composition
├── run_experiment.py          # Pipeline-based experiment runner
└── run_view_experiment.py         # Web visualization server
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

**Baseline Generation Framework** (unified):
- `baseline_utils.py` - Shared utilities for all baseline generators (JSONL caching, result formatting, core workflow)
- `generate_baseline.py` - **NEW** Unified baseline generation script supporting code, math, and science tasks

**Pipeline Experiment Framework**:
- `run_experiment.py` - Configurable pipeline for processing reasoning (mask, truncate, shuffle, insert) with automatic result caching and grading
- `run_view_experiment.py` - Web visualization server for browsing experiment results with tree structure and conditional probability analysis
- `pipeline.py` - Core pipeline infrastructure with processor composition
- Output structure: `experiments/exp/<processor1>/<processor2>/.../results.json`

## Datasets

- **CodeElo**: C++ competitive programming problems
  - Path: `datasets/CodeElo/data/test.json`
- **AIME2025**: American Invitational Mathematics Examination problems
  - Path: `datasets/AIME2025/data.json`
  - Supports repeated sampling with suffix (e.g., `AIME2025__R10` for 10 repetitions)
- **GPQA-Diamond** (NEW): Science multiple-choice questions (198 questions)
  - Path: `datasets/GPQA-Diamond/test/gpqa_diamond.parquet`
  - Format: Parquet with `question` and `answer` (A/B/C/D) columns

## Common Commands

### Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Baseline Generation Scripts

Unified baseline generation with support for code, math, and science tasks:

```bash
# Generate CodeElo baseline results
python generate_baseline.py --task_type code \
    --output_path data/CodeElo/gpt-oss/p1/results.json

# Generate AIME2025 baseline with R10 (10 repetitions)
python generate_baseline.py --task_type math \
    --repeat_num 10 \
    --output_path data/AIME2025__R10/gpt-oss/p1/results.json

# Generate GPQA-Diamond baseline (NEW)
python generate_baseline.py --task_type science \
    --output_path data/GPQA-Diamond/gpt-oss/p1/results.json

# Test with small limit
python generate_baseline.py --task_type math --limit 2
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
python run_view_experiment.py [--port 5000] [--host 127.0.0.1] [--exp-dir experiments/exp/]
```

## Key Design Patterns

1. **Concurrent Execution**: Uses `ThreadPoolExecutor` for large-scale parallel inference with configurable `CONCURRENCY` settings
2. **Incremental Saving**: JSONL format with automatic resume capability (checks existing results before re-running)
3. **Retry Mechanism**: `MAX_TRY` parameter for handling API failures
4. **Data Classes**: Uses `@dataclass` for clear data structures (Request, Response, Task)
5. **Modular Experiments**: Each experiment type has its own script sharing the core infrastructure (`llm_client.py` and `core.py`)
6. **DRY Principle**: Common utilities extracted to:
   - `core.py` - General experiment utilities
   - `baseline_utils.py` - Baseline generation utilities (refactored to support code, math, science tasks)
7. **Task-Specific Callbacks**: Unified baseline generation uses callbacks for prompt building and result formatting to support different task types without code duplication

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

experiments/                         # Experiment results root
├── exp/                             # Pipeline experiment results
│   └── [processor1]/                # e.g., mask_number
│       └── [processor2]/            # e.g., shuffle_line
│           └── .../                 # nested processors
│               ├── results.json     # Final results with metadata
│               ├── results_stage1.jsonl # Generation stage (cached)
│               └── results_stage2.jsonl # Grading stage (cached)
├── exp_AIME2025__R10_deepseek/      # Dataset-specific results
├── exp_CodeElo_gpt-oss/             # Dataset-specific results
└── exp_*/                           # Other dataset-specific results

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

4. **Visualization**: Use `run_view_experiment.py` to browse results with tree structure and conditional probability analysis

### Development Requirements

**IMPORTANT**: When adding new features or modifying existing functionality, you **MUST**:

1. **Add unit tests** in the `tests/` directory
2. **Update documentation** in the prefix docstring of `run_experiment.py` if the changes affect:
   - Available processors or their parameters
   - Processing modes or options
   - Usage examples
   - Pipeline configuration syntax

#### Testing Requirements

1. **Test Coverage Guidelines**:
   - Every new function or method should have at least one test
   - Critical path logic should have comprehensive test coverage
   - Edge cases and error handling should be tested
   - Integration tests for pipeline components should be included

2. **Test Organization**:
   - Test files should follow the naming convention: `test_*.py`
   - Test functions should follow the naming convention: `test_*`
   - Group related tests in the same test file
   - Use descriptive test names that clearly indicate what is being tested

3. **Running Tests**:
   ```bash
   # Run all tests
   source .venv/bin/activate && python -m pytest tests/

   # Run specific test file
   source .venv/bin/activate && python -m pytest tests/test_core.py

   # Run with verbose output
   source .venv/bin/activate && python -m pytest -v tests/

   # Run with coverage report
   source .venv/bin/activate && python -m pytest --cov=. tests/
   ```

4. **Test-Driven Development (TDD)**:
   - Follow the Red-Green-Refactor cycle when developing new features
   - Write tests first to define expected behavior
   - Implement the minimum code to make tests pass
   - Refactor code while ensuring tests continue to pass

5. **Existing Test Files**:
   - `test_core.py` - Tests for core utilities
   - `test_pipeline.py` - Tests for pipeline framework
   - `test_processors.py` - Tests for processing functions
   - `test_masking.py` - Tests for masking operations
   - `test_preprocessing.py` - Tests for preprocessing operations
   - `test_mask_advance.py` - Tests for advanced masking features
   - Integration tests for specific processors and workflows

#### Documentation Update Requirements

When adding or modifying processors, **you MUST update the prefix docstring in `run_experiment.py`**:

1. **For New Processors**:
   - Add the processor to the "Available Processors" section
   - Document all available modes with clear descriptions
   - List all parameters (required and optional) with defaults
   - Add at least one usage example showing the new processor

2. **For New Modes or Parameters**:
   - Add the mode to the appropriate processor's mode list
   - Explain what the mode does and when to use it
   - Document any new parameters with their types and defaults
   - Update examples if the new mode is commonly used

3. **Documentation Format**:
   The docstring follows this structure:
   ```
   Available Processors
   --------------------
   processor_name(mode, param1='default', ...)
       Brief description.

       Modes:
       - 'mode1': Description
       - 'mode2': Description

       Optional parameters:
       - param1: Description (default: value)

   Examples
   --------
   python run_experiment.py --flow "processor('mode',param=value)"
   ```

4. **Keep Documentation Synchronized**:
   - Documentation should match the actual implementation
   - Remove deprecated features from the documentation
   - Update examples when the API changes
