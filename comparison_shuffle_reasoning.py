"""
Comparison experiment: Normal reasoning vs Shuffled reasoning

This script compares model performance when:
1. Control group: Normal reasoning with reasoning:high mode
2. Experimental group: Shuffled reasoning (line-by-line shuffle of the reasoning process)

Uses transformers directly instead of vLLM to have full control over reasoning content.
"""

import json
import random
import os
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass, asdict
import time
import argparse

# from llm_client import LLMClient, Task, Request  # Not needed for inference-only mode

# Debug flag controlled by environment variable
DEBUG = os.environ.get('DEBUG', '0') == '1'

def debug_print(msg: str):
    """Print debug message if DEBUG is enabled"""
    if DEBUG:
        print(msg)


@dataclass
class ExperimentResult:
    """Result from a single question"""
    question_id: int
    question: str
    ground_truth: str

    # Full reasoning results (before truncation)
    full_reasoning: str
    full_answer: str
    full_generation_time: float

    # Normal reasoning results (after truncation)
    normal_reasoning: str
    normal_answer: str
    normal_generation_time: float

    # Shuffled reasoning results
    shuffled_reasoning: str
    shuffled_answer: str
    shuffled_generation_time: float

    # Evaluation metrics
    is_same_answer: bool = None
    is_full_correct: bool = None
    is_normal_correct: bool = None
    is_shuffle_correct: bool = None

    # Evaluation reasoning
    same_answer_reasoning: str = None
    full_correct_reasoning: str = None
    normal_correct_reasoning: str = None
    shuffle_correct_reasoning: str = None


class GPTOssInference:
    """Direct transformers inference for GPT-OSS model"""

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize the model with transformers

        Args:
            model_path: Path to the model directory
            device: Device to load model on
        """
        print(f"Loading model from {model_path}...")
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

        print(f"Model loaded successfully on {device}")

    def _build_prompt(self, question: str, system_prompt: str = "You are a helpful assistant.", reasoning_effort='medium') -> str:
        """Build the prompt for the model"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        # Use tokenizer's chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=reasoning_effort,
        )
        debug_print(f'\n[DEBUG] _build_prompt:\n{prompt}\n')
        return prompt

    def _build_prompt_with_reasoning(self, question: str, reasoning: str,
                                     system_prompt: str = "You are a helpful assistant.", reasoning_effort='medium') -> str:
        """
        Build prompt with pre-filled reasoning content

        The idea is to provide the shuffled reasoning as if the model already generated it,
        then let it continue to generate the answer.

        We need to construct the partial response in the format:
        <|channel|>analysis<|message|>{shuffled_reasoning}<|end|><|start|>assistant<|channel|>final<|message|>
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=reasoning_effort,
        )

        # Append the pre-filled reasoning in the correct format
        # The model should continue from here to generate the final answer
        prompt += f"<|channel|>analysis<|message|>{reasoning}<|end|><|start|>assistant<|channel|>final<|message|>"

        debug_print(f'\n[DEBUG] _build_prompt_with_reasoning:\n{prompt}\n')
        return prompt

    def generate(self, prompt: str, max_new_tokens: int = 10000,
                temperature: float = 0.6, top_p: float = 0.95) -> Tuple[str, float]:
        """
        Generate response from the model

        Returns:
            (generated_text, generation_time)
        """
        start_time = time.time()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=False
        )

        debug_print(f'\n[DEBUG] generate output:\n{generated_text}\n')

        generation_time = time.time() - start_time
        return generated_text, generation_time

    def generate_with_reasoning_extraction(self, question: str,
                                           system_prompt: str = "You are a helpful assistant.") -> Tuple[str, str, float]:
        """
        Generate answer with reasoning extraction

        For GPT-OSS, the model should naturally separate reasoning from answer.
        We'll need to parse the output to extract both parts.

        Returns:
            (reasoning_content, answer_content, generation_time)
        """
        prompt = self._build_prompt(question, system_prompt, reasoning_effort='medium')
        generated_text, gen_time = self.generate(prompt)

        # Parse reasoning and answer
        # GPT-OSS format may vary, we'll try to split intelligently
        reasoning, answer = self._parse_reasoning_and_answer(generated_text)

        # Post-process: clean multiple consecutive newlines
        reasoning = clean_multiple_newlines(reasoning)

        debug_print(f'\n[DEBUG] Parsed reasoning:\n{reasoning}\n')
        debug_print(f'\n[DEBUG] Parsed answer:\n{answer}\n')

        return reasoning, answer, gen_time

    def generate_with_prefilled_reasoning(self, question: str, reasoning: str,
                                         system_prompt: str = "You are a helpful assistant.") -> Tuple[str, float]:
        """
        Generate answer with pre-filled reasoning

        Args:
            question: The problem to solve
            reasoning: Pre-filled reasoning content (possibly shuffled)

        Returns:
            (answer_content, generation_time)
        """
        prompt = self._build_prompt_with_reasoning(question, reasoning, system_prompt, reasoning_effort='medium')
        generated_text, gen_time = self.generate(prompt, max_new_tokens=2048)

        return generated_text, gen_time

    def _parse_reasoning_and_answer(self, text: str) -> Tuple[str, str]:
        """
        Parse generated text into reasoning and answer parts

        GPT-OSS format:
        <|channel|>analysis<|message|>{REASONING-PROCESS}<|end|><|start|>assistant<|channel|>final<|message|>{FINAL-ANSWER}<|return|>
        """
        reasoning = ""
        answer = ""

        # Find analysis section
        if "<|channel|>analysis<|message|>" in text:
            analysis_start = text.find("<|channel|>analysis<|message|>") + len("<|channel|>analysis<|message|>")
            analysis_end = text.find("<|end|>", analysis_start)
            if analysis_end > analysis_start:
                reasoning = text[analysis_start:analysis_end].strip()
        else:
            raise Exception('"<|channel|>analysis<|message|>" not in generation')

        # Find final answer section
        if "<|channel|>final<|message|>" in text:
            final_start = text.find("<|channel|>final<|message|>") + len("<|channel|>final<|message|>")
            final_end = text.find("<|return|>", final_start)
            if final_end > final_start:
                answer = text[final_start:final_end].strip()
            elif final_end == -1:
                # No <|return|> found, take rest of text
                answer = text[final_start:].strip()
        else:
            raise Exception('"<|channel|>final<|message|>" not in generation')

        # If parsing failed, return the whole text
        if not reasoning and not answer:
            reasoning = text
            answer = text

        return reasoning, answer

    def _extract_boxed_answer(self, text: str) -> str:
        """Extract answer from \\boxed{} if present"""
        if "\\boxed{" in text:
            start = text.rfind("\\boxed{")
            # Find matching closing brace
            count = 0
            for i in range(start + 7, len(text)):
                if text[i] == '{':
                    count += 1
                elif text[i] == '}':
                    if count == 0:
                        return text[start:i+1]
                    count -= 1
        return text

    def evaluate_same_answer(self, answer1: str, answer2: str) -> Tuple[bool, str]:
        """
        Evaluate if two answers are the same

        Returns:
            (is_same, reasoning)
        """
        prompt = f"""Compare these two answers and determine if they are mathematically equivalent.

**Answer 1:**
{answer1}

**Answer 2:**
{answer2}

**Task:**
Determine if these two answers are the same or equivalent.

**Guidelines:**
- Consider mathematical equivalence (e.g., 1/2 = 0.5, 2x = x + x)
- Ignore formatting differences if the mathematical content is the same
- Answer with \\boxed{{YES}} if they are the same, or \\boxed{{NO}} if they are different

Provide your reasoning first, then give your final answer in \\boxed{{}}."""

        generated_text, _ = self.generate(
            self._build_prompt(prompt, "You are a helpful mathematical assistant.", reasoning_effort='low'),
            max_new_tokens=1024,
            temperature=0.01
        )

        # Parse the response
        is_same = 'YES' in generated_text.upper() and 'NO' not in generated_text.upper()
        if 'YES' in generated_text.upper() and 'NO' in generated_text.upper():
            # Both present, need to check which comes last in boxed format
            yes_pos = generated_text.upper().rfind('\\BOXED{YES}')
            no_pos = generated_text.upper().rfind('\\BOXED{NO}')
            is_same = yes_pos > no_pos

        return is_same, generated_text

    def evaluate_correctness(self, problem: str, ground_truth: str, model_answer: str) -> Tuple[bool, str]:
        """
        Evaluate if model's answer is correct

        Returns:
            (is_correct, reasoning)
        """
        prompt = f"""**Problem:**
{problem}

**Ground Truth Answer:**
{ground_truth}

**Model's Answer:**
{model_answer}

**Task: Grading**
Please determine if the model's answer is correct compared to the ground truth answer.

**Guidelines:**
- Consider mathematical equivalence (e.g., 1/2 = 0.5, 2x = x + x)
- Ignore formatting differences if the mathematical content is the same
- Answer with \\boxed{{YES}} if correct, or \\boxed{{NO}} if incorrect

Provide your reasoning first, then give your final answer in \\boxed{{}}."""

        generated_text, _ = self.generate(
            self._build_prompt(prompt, "You are a helpful mathematical grading assistant.", reasoning_effort='low'),
            max_new_tokens=1024,
            temperature=0.01
        )

        # Parse the response
        is_correct = 'YES' in generated_text.upper() and 'NO' not in generated_text.upper()
        if 'YES' in generated_text.upper() and 'NO' in generated_text.upper():
            # Both present, need to check which comes last in boxed format
            yes_pos = generated_text.upper().rfind('\\BOXED{YES}')
            no_pos = generated_text.upper().rfind('\\BOXED{NO}')
            is_correct = yes_pos > no_pos

        return is_correct, generated_text


def clean_multiple_newlines(text: str) -> str:
    """
    Replace multiple consecutive newlines with a single newline

    Args:
        text: Input text with potentially multiple consecutive newlines

    Returns:
        Cleaned text with at most one newline between lines
    """
    import re
    # Replace 2 or more consecutive newlines with a single newline
    cleaned = re.sub(r'\n\n+', '\n', text)
    return cleaned


def truncate_reasoning_lines(reasoning: str, del_last_line_count: int) -> str:
    """
    Remove last n lines from reasoning content

    Args:
        reasoning: Original reasoning content
        del_last_line_count: Number of lines to remove from the end

    Returns:
        Truncated reasoning content
    """
    if del_last_line_count <= 0:
        return reasoning

    lines = reasoning.strip().split('\n')
    # Remove empty lines
    lines = [line for line in lines if line.strip()]

    # Remove last n lines
    if del_last_line_count >= len(lines):
        # If trying to remove more lines than available, return empty string or first line
        return lines[0] if lines else ""

    truncated_lines = lines[:-del_last_line_count]
    return '\n'.join(truncated_lines)


def shuffle_reasoning_lines(reasoning: str) -> str:
    """
    Shuffle reasoning content line-by-line

    Args:
        reasoning: Original reasoning content

    Returns:
        Shuffled reasoning content
    """
    lines = reasoning.strip().split('\n')
    # Remove empty lines
    lines = [line for line in lines if line.strip()]

    # Shuffle
    random.shuffle(lines)

    return '\n'.join(lines)


def load_dataset(dataset_path: str) -> List[Dict]:
    """Load AIME2025 dataset"""
    problems = []
    with open(dataset_path, 'r') as f:
        for line in f:
            problems.append(json.loads(line))
    return problems


def run_experiment(model_path: str, dataset_path: str, output_dir: str,
                  num_samples: int = None, seed: int = 42, del_last_line_count: int = 0):
    """
    Run the comparison experiment

    Args:
        model_path: Path to GPT-OSS model
        dataset_path: Path to AIME2025 dataset
        output_dir: Directory to save results
        num_samples: Number of samples to test (None = all)
        seed: Random seed for shuffling
        del_last_line_count: Number of lines to remove from end of reasoning (default: 0)
    """
    random.seed(seed)
    torch.manual_seed(seed)

    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = GPTOssInference(model_path)

    # Load dataset
    problems = load_dataset(dataset_path)
    if num_samples:
        problems = problems[:num_samples]

    print(f"\nRunning experiment on {len(problems)} problems...")

    # Store results
    results = []

    # Process each problem
    for idx, problem in enumerate(tqdm(problems, desc="Processing problems")):
        question = problem['question']
        ground_truth = problem['answer']

        try:
            # Step 1: Generate with full reasoning (initial generation)
            print(f"\n[{idx+1}/{len(problems)}] Generating full reasoning...")
            full_reasoning, full_answer, full_time = model.generate_with_reasoning_extraction(question)

            # Step 2: Truncate reasoning (remove last n lines)
            normal_reasoning = truncate_reasoning_lines(full_reasoning, del_last_line_count)
            debug_print(f'\n[DEBUG] Full reasoning has {len(full_reasoning.split(chr(10)))} lines')
            debug_print(f'[DEBUG] Truncated reasoning has {len(normal_reasoning.split(chr(10)))} lines')
            debug_print(f'[DEBUG] Truncated reasoning:\n{normal_reasoning}\n')

            # Step 3: Generate answer with truncated normal reasoning
            print(f"[{idx+1}/{len(problems)}] Generating with truncated normal reasoning...")
            normal_answer, normal_time = model.generate_with_prefilled_reasoning(
                question, normal_reasoning
            )

            # Step 4: Shuffle the truncated reasoning
            shuffled_reasoning = shuffle_reasoning_lines(normal_reasoning)
            debug_print(f'\n[DEBUG] Shuffled reasoning:\n{shuffled_reasoning}\n')

            # Step 5: Generate answer with shuffled reasoning
            print(f"[{idx+1}/{len(problems)}] Generating with shuffled reasoning...")
            shuffled_answer, shuffled_time = model.generate_with_prefilled_reasoning(
                question, shuffled_reasoning
            )

            # Step 6: Evaluate the results
            print(f"[{idx+1}/{len(problems)}] Evaluating results...")

            # Check if full answer is correct
            is_full_correct, full_correct_reasoning = model.evaluate_correctness(
                question, ground_truth, full_answer
            )
            print(f"  - is_full_correct: {is_full_correct}")

            # Check if normal answer is correct
            is_normal_correct, normal_correct_reasoning = model.evaluate_correctness(
                question, ground_truth, normal_answer
            )
            print(f"  - is_normal_correct: {is_normal_correct}")

            # Check if shuffled answer is correct
            is_shuffle_correct, shuffle_correct_reasoning = model.evaluate_correctness(
                question, ground_truth, shuffled_answer
            )
            print(f"  - is_shuffle_correct: {is_shuffle_correct}")

            # Check if answers are the same
            # New rule: If both normal and shuffle are correct, they must be the same
            if is_normal_correct and is_shuffle_correct:
                is_same = True
                same_reasoning = "Both answers are correct, therefore they must be the same."
                print(f"  - is_same_answer: {is_same} (auto-set: both correct)")
            else:
                is_same, same_reasoning = model.evaluate_same_answer(normal_answer, shuffled_answer)
                print(f"  - is_same_answer: {is_same}")

            # Store result
            result = ExperimentResult(
                question_id=idx,
                question=question,
                ground_truth=ground_truth,
                full_reasoning=full_reasoning,
                full_answer=full_answer,
                full_generation_time=full_time,
                normal_reasoning=normal_reasoning,
                normal_answer=normal_answer,
                normal_generation_time=normal_time,
                shuffled_reasoning=shuffled_reasoning,
                shuffled_answer=shuffled_answer,
                shuffled_generation_time=shuffled_time,
                is_same_answer=is_same,
                is_full_correct=is_full_correct,
                is_normal_correct=is_normal_correct,
                is_shuffle_correct=is_shuffle_correct,
                same_answer_reasoning=same_reasoning,
                full_correct_reasoning=full_correct_reasoning,
                normal_correct_reasoning=normal_correct_reasoning,
                shuffle_correct_reasoning=shuffle_correct_reasoning
            )
            results.append(result)

            # Save incrementally
            results_path = output_dir / "experiment_results.json"
            with open(results_path, 'w') as f:
                json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error processing problem {idx}: {e}")
            continue

    # Calculate summary statistics
    total = len(results)
    same_answer_count = sum(1 for r in results if r.is_same_answer)
    full_correct_count = sum(1 for r in results if r.is_full_correct)
    normal_correct_count = sum(1 for r in results if r.is_normal_correct)
    shuffle_correct_count = sum(1 for r in results if r.is_shuffle_correct)

    summary = {
        'total_problems': total,
        'del_last_line_count': del_last_line_count,
        'same_answer': {
            'count': same_answer_count,
            'percentage': round(same_answer_count / total * 100, 2) if total > 0 else 0
        },
        'full_correct': {
            'count': full_correct_count,
            'percentage': round(full_correct_count / total * 100, 2) if total > 0 else 0
        },
        'normal_correct': {
            'count': normal_correct_count,
            'percentage': round(normal_correct_count / total * 100, 2) if total > 0 else 0
        },
        'shuffle_correct': {
            'count': shuffle_correct_count,
            'percentage': round(shuffle_correct_count / total * 100, 2) if total > 0 else 0
        },
        'difference': {
            'normal_vs_shuffle': {
                'absolute': normal_correct_count - shuffle_correct_count,
                'percentage': round((normal_correct_count - shuffle_correct_count) / total * 100, 2) if total > 0 else 0
            },
            'full_vs_normal': {
                'absolute': full_correct_count - normal_correct_count,
                'percentage': round((full_correct_count - normal_correct_count) / total * 100, 2) if total > 0 else 0
            },
            'full_vs_shuffle': {
                'absolute': full_correct_count - shuffle_correct_count,
                'percentage': round((full_correct_count - shuffle_correct_count) / total * 100, 2) if total > 0 else 0
            }
        }
    }

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Experiment complete. Results saved to {output_dir}")
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total problems: {total}")
    print(f"Deleted last {del_last_line_count} lines from reasoning")
    print(f"\nSame answer: {same_answer_count}/{total} ({summary['same_answer']['percentage']:.2f}%)")
    print(f"\nCorrectness:")
    print(f"  Full correct:    {full_correct_count}/{total} ({summary['full_correct']['percentage']:.2f}%)")
    print(f"  Normal correct:  {normal_correct_count}/{total} ({summary['normal_correct']['percentage']:.2f}%)")
    print(f"  Shuffle correct: {shuffle_correct_count}/{total} ({summary['shuffle_correct']['percentage']:.2f}%)")
    print(f"\nDifferences:")
    print(f"  Normal vs Shuffle: {summary['difference']['normal_vs_shuffle']['absolute']} ({summary['difference']['normal_vs_shuffle']['percentage']:+.2f}%)")
    print(f"  Full vs Normal:    {summary['difference']['full_vs_normal']['absolute']} ({summary['difference']['full_vs_normal']['percentage']:+.2f}%)")
    print(f"  Full vs Shuffle:   {summary['difference']['full_vs_shuffle']['absolute']} ({summary['difference']['full_vs_shuffle']['percentage']:+.2f}%)")
    print(f"{'='*80}")

    return results


def grade_results(results: List[ExperimentResult], output_dir: Path,
                 judge_model_type: str = 'deepseek'):
    """
    Grade both normal and shuffled results using the grading system

    Args:
        results: List of experiment results
        output_dir: Output directory
        judge_model_type: Model to use for grading
    """
    print("\n" + "="*80)
    print("GRADING RESULTS")
    print("="*80)

    client = LLMClient()

    # Prepare grading tasks for both conditions
    normal_tasks = []
    shuffled_tasks = []

    GRADING_PROMPT = """
**Problem:**
{problem}

**Ground Truth Answer:**
{ground_truth}

**Model's Answer:**
{model_answer}

**Task: Grading**
Please determine if the model's answer is correct compared to the ground truth answer.

**Guidelines:**
- Consider mathematical equivalence (e.g., 1/2 = 0.5, 2x = x + x)
- Ignore formatting differences if the mathematical content is the same
- Answer with \\boxed{{YES}} if correct, or \\boxed{{NO}} if incorrect
"""

    for result in results:
        # Normal reasoning grading task
        normal_tasks.append(Task(
            index=result.question_id,
            request=Request(
                queries=[GRADING_PROMPT.format(
                    problem=result.question,
                    ground_truth=result.ground_truth,
                    model_answer=result.normal_answer
                )],
                model_type=judge_model_type,
                system_prompt="You are a helpful assistant",
                reasoning_on=False
            ),
            metadata={'condition': 'normal', 'result': result}
        ))

        # Shuffled reasoning grading task
        shuffled_tasks.append(Task(
            index=result.question_id,
            request=Request(
                queries=[GRADING_PROMPT.format(
                    problem=result.question,
                    ground_truth=result.ground_truth,
                    model_answer=result.shuffled_answer
                )],
                model_type=judge_model_type,
                system_prompt="You are a helpful assistant",
                reasoning_on=False
            ),
            metadata={'condition': 'shuffled', 'result': result}
        ))

    # Grade both conditions
    normal_grades = []
    shuffled_grades = []

    print("\nGrading normal reasoning answers...")
    for task in tqdm(client.generate_concurrent(normal_tasks, max_workers=50),
                     total=len(normal_tasks)):
        if task.response.success:
            is_correct = 'YES' in task.response.content and 'NO' not in task.response.content
            normal_grades.append({
                'question_id': task.index,
                'correct': is_correct,
                'grading_response': task.response.content
            })

    print("\nGrading shuffled reasoning answers...")
    for task in tqdm(client.generate_concurrent(shuffled_tasks, max_workers=50),
                     total=len(shuffled_tasks)):
        if task.response.success:
            is_correct = 'YES' in task.response.content and 'NO' not in task.response.content
            shuffled_grades.append({
                'question_id': task.index,
                'correct': is_correct,
                'grading_response': task.response.content
            })

    # Calculate statistics
    normal_correct = sum(g['correct'] for g in normal_grades)
    shuffled_correct = sum(g['correct'] for g in shuffled_grades)
    total = len(normal_grades)

    summary = {
        'total_problems': total,
        'normal_reasoning': {
            'correct': normal_correct,
            'incorrect': total - normal_correct,
            'accuracy': round(normal_correct / total * 100, 2) if total > 0 else 0
        },
        'shuffled_reasoning': {
            'correct': shuffled_correct,
            'incorrect': total - shuffled_correct,
            'accuracy': round(shuffled_correct / total * 100, 2) if total > 0 else 0
        },
        'difference': {
            'absolute': normal_correct - shuffled_correct,
            'percentage': round((normal_correct - shuffled_correct) / total * 100, 2) if total > 0 else 0
        }
    }

    # Save results
    grading_results = {
        'summary': summary,
        'normal_grades': normal_grades,
        'shuffled_grades': shuffled_grades
    }

    grades_path = output_dir / "grading_results.json"
    with open(grades_path, 'w') as f:
        json.dump(grading_results, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("GRADING SUMMARY")
    print("="*80)
    print(f"\nTotal problems: {total}")
    print(f"\nNormal reasoning:")
    print(f"  Correct: {normal_correct}/{total} ({summary['normal_reasoning']['accuracy']:.2f}%)")
    print(f"\nShuffled reasoning:")
    print(f"  Correct: {shuffled_correct}/{total} ({summary['shuffled_reasoning']['accuracy']:.2f}%)")
    print(f"\nDifference:")
    print(f"  {summary['difference']['absolute']} problems ({summary['difference']['percentage']:+.2f}%)")
    print(f"\nResults saved to {grades_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare normal vs shuffled reasoning")
    parser.add_argument("--model_path", type=str,
                       default="/mnt/shared/p01/yc/models/gpt-oss-120b",
                       help="Path to GPT-OSS model")
    parser.add_argument("--dataset_path", type=str,
                       default="/mnt/shared/p01/yc/datasets/AIME2025/aime2025-I.jsonl",
                       help="Path to AIME2025 dataset")
    parser.add_argument("--output_dir", type=str,
                       default="./data/shuffle_reasoning_comparison",
                       help="Output directory for results")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to test (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--del_last_line_count", type=int, default=0,
                       help="Number of lines to remove from end of reasoning before comparison (default: 0)")

    args = parser.parse_args()

    # Run experiment
    results = run_experiment(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        del_last_line_count=args.del_last_line_count
    )

    print(f"\n✓ Experiment complete!")
    print(f"✓ Results saved to {args.output_dir}/experiment_results.json")
    print(f"✓ Total problems processed: {len(results)}")


if __name__ == "__main__":
    main()
