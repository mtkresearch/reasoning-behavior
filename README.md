# Rethinking Dense Sequential Chains

> **Reasoning Language Models Can Extract Answers from Sparse, Order-Shuffling Chain-of-Thoughts**

This repository contains code and experiment scripts accompanying the paper. It demonstrates that RLMs can accurately extract answers from drastically transformed reasoning chains — shuffled, masked, or stripped to a sparse numeric skeleton — revealing that sequential ordering and dense prose are largely dispensable.

**Authors:** Yi-Chang Chen, Feng-Ting Liao, Da-shan Shiu (MediaTek Research), Hung-yi Lee (National Taiwan University)

---

## Key Findings

Experiments span three models (GPT-OSS-120B, DeepSeek-V3.1-671B, OLMo-3.1-32B) and three benchmarks (AIME 2025, CodeElo, GPQA-Diamond). We evaluate under two modes: free **Generation (Gen)** and **Retrieval (Ret)**. Ret is adopted as the primary setting — it forces the model to complete a fixed answer-extraction prefix, ensuring all measured accuracy is attributable to the structure of the reasoning chain itself.

**Finding 1 — Generation mode implicitly re-reasons when the chain is absent or degraded — making retrieval mode the appropriate mode for isolating extraction.**
In Gen mode, models can re-solve the problem independently when the chain is absent or degraded (e.g., OLMo reaches 28.33% on AIME 2025 *without* any chain). This conflates extraction with native problem-solving. In Ret mode, accuracy collapses to 0.00% under Reasoning-Free, cleanly isolating what the chain contributes.

**Finding 2 — Reasoning chains are order-independent at the line level, and shuffled chains provide genuine extractable signal.**
Randomly permuting all lines (Line-Shuffle) causes negligible accuracy loss: OLMo is unchanged at 78.00%, GPT-OSS drops only 0.39 pp (91.33% → 90.94%), and DeepSeek shows a slight improvement on AIME 2025. The sequential ordering of reasoning steps is almost entirely unnecessary for answer extraction.

**Finding 3 — Word-level semantic units are the minimum necessary granularity; once this minimum meaning-conveying unit is intact, ordering above it has limited impact on extraction.**
Word-level shuffling (all words globally scrambled) still yields 62–89% accuracy — far above the reasoning-free floor of 0%. Token-level shuffling collapses to 1–3%, approaching the same floor. The Random-Word baseline (replacing words with same-frequency random words) also scores near 0%, confirming that *word identity*, not just word-level ordering, is what provides signal.

**Finding 4 — Order-independence is present in pretrained models, suggesting it originates from pretraining rather than reasoning-specific fine-tuning.**
OLMo-3.1-32B-Base (pretrained only) and OLMo-3.1-32B (instruction-tuned) achieve nearly identical accuracy under Line-Shuffle (78.67% vs. 78.00%) and Word-Shuffle (78.33% vs. 77.67%) on AIME 2025. Reasoning-specific fine-tuning confers no additional robustness to disordered chains.

**Finding 5 — Informational content is non-uniformly distributed — alphabetic prose is redundant, the explicit answer occurrence is a strong extraction anchor, and numeric content is irreducible.**
Masking all alphabetic characters *increases* GPT-OSS accuracy by 4.7 pp (91.3% → 96.0%) — prose is redundant. Masking only the answer token(s) causes an 18.0 pp drop (91.3% → 73.3%) — the explicit answer is a primary extraction anchor. Masking all digits collapses accuracy to 0.0% in every condition — numeric content is irreducible.

**Finding 6 — The non-uniformly distributed informational content is sufficient on its own — perturbing its positional arrangement causes only minimal signal loss.**
Removing all alphabetic text and then shuffling lines (Remove-Alphabet + Line-Shuffle) yields 83.3%, only −8.0 pp from the unperturbed baseline. A chain stripped of all natural language and presented in arbitrary line order still extracts the correct answer 83% of the time. Removing the question prompt costs at most 2.0 pp.

**Finding 7 — The structural signal identified in Finding 6 is intrinsically robust — even strong false-answer injection cannot override it.**
Injecting the spurious sentence "Thus answer: 123." at 1×, 2×, and 3× the true-answer frequency leaves accuracy essentially unchanged across all representation conditions. Even the most reduced representation (line-shuffled + alphabet-removed) stays flat at 83.3% regardless of noise multiplier. This falsifies frequency-based extraction — the model is highly sensitive to the *absence* of the correct answer yet entirely insensitive to a *surplus* of competing false ones.

**Takeaway:** Transformers appear to process reasoning chains as approximately unordered sets of symbolic constraints, attending selectively to numeric values and relational structures. *What* symbolic content is present matters; *where* it appears largely does not.

---

## Setup

**1. Create virtual environment and install dependencies**

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

**2. Configure OpenRouter API key**

All experiments run through [OpenRouter](https://openrouter.ai/). Create a `.env` file in the project root:

```bash
OPENROUTER_API_KEY=your_key_here
```

---

## Datasets

| Dataset | Task | Path | Size |
|---|---|---|---|
| AIME 2025 | Math (competition) | `datasets/AIME2025/data.json` | 30 problems × 10 samples |
| CodeElo | Code (competitive programming) | `datasets/CodeElo/data/test.json` | 408 problems |
| GPQA-Diamond | Science (multiple choice) | `datasets/GPQA-Diamond/test/gpqa_diamond.parquet` | 198 questions |

---

## Reproducing Paper Experiments

All experiments follow two steps: generate baseline reasoning chains, then apply transformations and evaluate.

### Step 1 — Generate Baselines

Run once per model × dataset combination. Results are saved under `data/`.

```bash
# AIME 2025 — GPT-OSS (10 repeated samples per problem)
python generate_baseline.py --task_type math --model_type gpt-oss --repeat_num 10 \
    --output_path data/AIME2025__R10/gpt-oss/p1/results.json

# CodeElo — GPT-OSS
python generate_baseline.py --task_type code --model_type gpt-oss \
    --output_path data/CodeElo/gpt-oss/p1/results.json

# GPQA-Diamond — GPT-OSS
python generate_baseline.py --task_type science --model_type gpt-oss \
    --output_path data/GPQA-Diamond/gpt-oss/p1/results.json
```

Replace `--model_type gpt-oss` with `deepseek` or `olmo` for the other models in the paper.

> **Quick test:** Add `--limit 2` to verify the pipeline with 2 samples before a full run.

---

### Step 2 — Run Transformations

Use `run_experiment.py` with a `--flow` string that chains processors. The `--results_path` is the baseline generated in Step 1. Output is auto-saved under `experiments/exp/`.

```bash
python run_experiment.py \
    --flow "<processor_chain>" \
    --results_path data/AIME2025__R10/gpt-oss/p1/results.json
```

The sections below map each paper table to the corresponding `--flow` arguments.

---

### Table 2 — Shuffle Comparison

The paper's primary evaluation is **Retrieval (Ret) mode**: append `answer('retrieval')` to every flow so the model completes a fixed prefix ("Thus, the answer is …") rather than freely re-solving.

| Paper Condition | `--flow` |
|---|---|
| Baseline (Ret) | `answer('retrieval')` |
| Line-Shuffle ($\mathcal{S}_\text{line}$) | `shuffle('line'),answer('retrieval')` |
| Word-Shuffle ($\mathcal{S}_\text{word}$) | `shuffle('word'),answer('retrieval')` |
| In-line-Word-Shuffle ($\mathcal{S}_\text{ilw}$) | `shuffle('in-line-word'),answer('retrieval')` |
| Token-Shuffle ($\mathcal{S}_\text{tok}$) | `shuffle('token'),answer('retrieval')` |
| Random-Word ($\mathcal{D}_\text{word}$) | `padding('word',words_tsv_path='data/AIME2025__R10/gpt-oss/p1/words.tsv'),answer('retrieval')` |
| Random-Token ($\mathcal{D}_\text{tok}$) | `padding('token'),answer('retrieval')` |
| Reasoning-Free ($\mathcal{R}_r$) | `truncate('all'),answer('retrieval')` |

Example:

```bash
python run_experiment.py \
    --flow "shuffle('line'),answer('retrieval')" \
    --results_path data/AIME2025__R10/gpt-oss/p1/results.json
```

---

### Table 3 — Base Model Comparison (OLMo Instruct vs. Base)

Same flows as Table 2, but the baseline must be generated with the OLMo base model on a local VLLM server (not available on OpenRouter). Set `--mode local` and `--model_type olmo--base`.

---

### Tables 4–6 — Intervention & Ablation Studies

| Paper Condition | `--flow` |
|---|---|
| Mask-Alphabet ($\mathcal{M}_\alpha$) | `mask('alphabet'),answer('retrieval')` |
| Mask-Number ($\mathcal{M}_\nu$) | `mask('number'),answer('retrieval')` |
| Mask-Answer ($\mathcal{M}_\text{ans}$) | `mask('answer'),answer('retrieval')` |
| Remove-Alphabet ($\mathcal{R}_\alpha$) | `mask('alphabet',mask_char=' '),replace('\\s+',replacement=' '),answer('retrieval')` |
| Remove-Question ($\mathcal{R}_q$) | `question('remove'),answer('retrieval')` |
| Noise-Injection 1× ($\mathcal{N}_1$) | `insert('fix',sentence='Thus answer: {ANSWER}.',count='100% # of answer'),answer('retrieval')` |
| Remove-Alphabet + Line-Shuffle | `mask('alphabet',mask_char=' '),replace('\\s+',replacement=' '),shuffle('line'),answer('retrieval')` |

Processors can be chained freely. For example, the most reduced condition in Table 6 (alphabet removed + line-shuffled + 3× noise):

```bash
python run_experiment.py \
    --flow "mask('alphabet',mask_char=' '),replace('\\s+',replacement=' '),shuffle('line'),insert('fix',sentence='Thus answer: {ANSWER}.',count='300% # of answer'),answer('retrieval')" \
    --results_path data/AIME2025__R10/gpt-oss/p1/results.json
```

---

### Step 3 — View Results

```bash
python run_view_experiment.py
# Open http://localhost:5000
```

The web interface shows a tree of all experiments under `experiments/exp/` with accuracy metrics and conditional probability analysis.

---

## Citation

```bibtex
@article{chen2025rethinking,
  title   = {Rethinking Dense Sequential Chains: Reasoning Language Models Can Extract Answers from Sparse, Order-Shuffling Chain-of-Thoughts},
  author  = {Yi-Chang Chen and Feng-Ting Liao and Da-shan Shiu and Hung-yi Lee},
  year    = {2026},
}
```
