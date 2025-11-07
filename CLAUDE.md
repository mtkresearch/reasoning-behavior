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
├── data/                       # Experimental results
│   ├── [dataset]__[R{n}]/     # Main results
│   ├── shuffle_comparison_exp/ # Shuffle experiments
│   └── consistency_data/       # Consistency analysis
│
├── llm_client.py              # Core LLM abstraction
├── comparison_shuffle_reasoning.py  # Experiment: shuffle comparison
├── insert_noise_reasoning.py        # Experiment: noise insertion
├── insert_crossreasoning_noise.py   # Experiment: cross-reasoning
├── no_reasoning_baseline.py         # Experiment: no-reasoning baseline
├── analyze_shuffle_results.py       # Analysis utility
├── filter_by_consensus.py           # Filtering utility
└── run_shuffle_reasoning.sh         # Batch experiment runner
```

## Core Architecture

### Project Structure

The codebase is organized into two main layers:

**Root Directory** - Core infrastructure and experiment scripts:
- `llm_client.py` - Central LLM client abstraction
- Experiment scripts: `comparison_shuffle_reasoning.py`, `insert_noise_reasoning.py`, etc.
- Analysis utilities: `analyze_shuffle_results.py`, `filter_by_consensus.py`, etc.

### LLM Client Layer (`llm_client.py`)

The central abstraction for all LLM interactions:
- Supports both OpenRouter (remote) and local VLLM servers
- Provides synchronous and concurrent generation interfaces
- Supports multiple models: DeepSeek, GPT-OSS, Qwen3
- Key classes: `Request`, `Response`, `Task`, `LLMClient`

### Main Components

**Experiment Scripts** (in root directory):
- `comparison_shuffle_reasoning.py` - Normal vs shuffled reasoning
- `insert_noise_reasoning.py` - Noise insertion robustness test
- `insert_crossreasoning_noise.py` - Cross-question reasoning noise
- `no_reasoning_baseline.py` - No-reasoning baseline

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
# Shuffle reasoning comparison (supports multiple truncation ratios and methods)
bash run_shuffle_reasoning.sh

# Noise insertion experiments
python insert_noise_reasoning.py [target] [model_type] [system_type]
python insert_crossreasoning_noise.py [target] [model_type] [system_type]

# No-reasoning baseline
python no_reasoning_baseline.py [target] [model_type] [system_type]

# Analyze shuffle results
python analyze_shuffle_results.py [data_folder]
```

## Key Design Patterns

1. **Concurrent Execution**: Uses `ThreadPoolExecutor` for large-scale parallel inference with configurable `CONCURRENCY` settings
2. **Incremental Saving**: JSONL format with automatic resume capability (checks existing results before re-running)
3. **Retry Mechanism**: `MAX_TRY` parameter for handling API failures
4. **Data Classes**: Uses `@dataclass` for clear data structures (Request, Response, Task)
5. **Modular Experiments**: Each experiment type has its own script sharing the core `llm_client`

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

The project uses a hierarchical structure for organizing experimental data:

### Main Experiment Results

```
data/
├── [dataset]__[R{repetitions}]/     # e.g., AIME2025__R10
│   ├── [model]/                     # e.g., gpt-oss, deepseek, qwen3
│   │   ├── [prompt]/                # e.g., p1, p2, p3
│   │   │   ├── results.json         # Inference results
│   │   │   ├── grades.json          # Grading results
│   │   │   ├── metrics.json         # Evaluation metrics
│   │   │   ├── behavior.jsonl       # Extracted behaviors
│   │   │   ├── summary.pdf          # Generated report
│   │   │   ├── no_reasoning.json    # No-reasoning baseline
│   │   │   ├── insert-noise-*.json  # Noise insertion experiments
│   │   │   └── shuffle-insert-*.json # Shuffle+noise experiments
│   │
│   └── MATH500/                     # MATH500 dataset results
│       ├── deepseek/
│       ├── gpt-oss/
│       └── qwen3/
```

### Specialized Experiment Results

```
data/
├── shuffle_comparison_exp/          # Shuffle and truncation experiments
│   ├── empty_question_f00/          # Empty question test
│   ├── token_turncate_f00/          # Token-level truncation
│   ├── word_turncate_f00/           # Word-level truncation
│   ├── turncate_0/                  # 0 count truncation (baseline)
│   ├── turncate_3/                  # 3 count truncation
│   ├── turncate_5/                  # 5 count truncation
│   ├── turncate_7/                  # 7 count truncation
│   ├── turncate_9/                  # 9 count truncation
│   ├── turncate_f01/                # 10% truncation
│   ├── turncate_f03/                # 30% truncation (float)
│   ├── turncate_f05/                # 50% truncation (float)
│   ├── turncate_f07/                # 70% truncation (float)
│   └── turncate_f09/                # 90% truncation (float)
│
└── consistency_data/                # Consistency analysis results
    ├── consistency_data.json
    ├── consistency_deepseek.json
    ├── consistency_gpt-oss.json
    ├── consistency_gpt-oss-reasoning.json
    └── consistency_gpt5.json
```

### Dataset Sources

```
datasets/
└── AIME2025/                        # AIME 2025 problems
    └── [problem files...]
```

## Working with This Codebase

### Code Organization Principles

1. **Two-layer architecture**:
   - **Root directory**: Specialized experiment scripts that use the core pipeline
   - **behavior_analysis/**: Core pipeline tools and advanced analysis modules (deprecated)

2. **Standalone scripts**: Each Python file is designed to be run independently with clear command-line interfaces

3. **Results are cached**: All scripts check for existing results and skip completed work to support resumable execution

4. **No unit tests**: This is a research codebase; validation happens through experimental results and comparative analysis

5. **Incremental development**: Experiments build on each other; check related scripts before adding new ones

6. **Token-level operations**: Some experiments use transformers tokenizer for fine-grained text manipulation (shuffling, truncation at token boundaries)
