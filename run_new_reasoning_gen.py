#!/usr/bin/env python3
"""
Filtered Reasoning Generation Experiment

This script generates reasoning for AIME problems with token-level filtering:
- Stage 1: Generate reasoning while forbidding purely alphabetic tokens
- Stage 2: Generate answer using the generated reasoning

The script prevents purely alphabetic tokens (e.g., 'hello', 'world') but allows
mixed tokens (e.g., '3x', 'x_1'). Generation stops when <|end|> token is detected
or max_tokens_in_reasoning is reached.

=============================================================================
EXPERIMENT DESIGN
=============================================================================

This experiment tests whether model reasoning relies heavily on natural language
(alphabetic tokens) or primarily uses numbers and symbols. By forbidding purely
alphabetic tokens during reasoning generation, we force the model to express
reasoning patterns using only non-letter information.

Key Components:

1. LetterOnlyTokenFilter (LogitsProcessor):
   - Identifies all purely alphabetic tokens in the tokenizer vocabulary
   - During generation, sets logits of pure-letter tokens to -inf
   - Allows mixed tokens (letters+numbers) like '3x' or 'x_1'
   - Uses regex ^[A-Za-z]+$ to identify pure letters

2. EndTokenStoppingCriteria (StoppingCriteria):
   - Detects <|end|> token to mark end of reasoning section
   - Enables natural stopping point for reasoning generation
   - Prevents runaway generation

3. Two-Stage Generation:
   - Stage 1: Generate reasoning with token filtering
   - Stage 2: Generate answer using prefill guidance ("Thus, the answer is")

=============================================================================
DATA FORMAT
=============================================================================

Input: data/AIME2025__R10/gpt-oss/p1/results.json
  Fields used:
  - unique_id: Problem identifier
  - question: The problem statement
  - answer: Ground truth answer
  - result.sys_prompt: System prompt for generation

Output: JSON file with fields:
  - unique_id: Problem identifier
  - question: The problem statement
  - answer: Ground truth answer (for reference)
  - sys_prompt: System prompt used
  - reason_text: Generated reasoning (with token filtering applied)
  - answer_text: Generated answer text
  - error (optional): Error message if processing failed

=============================================================================
USAGE EXAMPLES
=============================================================================

1. Basic usage (test with 5 items):
   python run_new_reasoning_gen.py \
       --input_path data/AIME2025__R10/gpt-oss/p1/results.json \
       --output_path output/test_results.json \
       --model_path openai/gpt-oss-120b \
       --device cuda \
       --limit 5

2. Full run:
   python run_new_reasoning_gen.py \
       --input_path data/AIME2025__R10/gpt-oss/p1/results.json \
       --output_path output/filtered_reasoning_results.json \
       --model_path openai/gpt-oss-120b \
       --device cuda

3. Custom max reasoning tokens:
   python run_new_reasoning_gen.py \
       --input_path data/AIME2025__R10/gpt-oss/p1/results.json \
       --output_path output/short_reasoning_results.json \
       --model_path openai/gpt-oss-120b \
       --device cuda \
       --max_reasoning_tokens 1000

4. CPU inference:
   python run_new_reasoning_gen.py \
       --input_path data/AIME2025__R10/gpt-oss/p1/results.json \
       --output_path output/cpu_results.json \
       --model_path openai/gpt-oss-120b \
       --device cpu

5. Debug logging:
   python run_new_reasoning_gen.py \
       --input_path data/AIME2025__R10/gpt-oss/p1/results.json \
       --output_path output/debug_results.json \
       --model_path openai/gpt-oss-120b \
       --device cuda \
       --log_level DEBUG \
       --limit 2

=============================================================================
COMMAND LINE ARGUMENTS
=============================================================================

--input_path INPUT_PATH
    Path to input results.json file
    Default: data/AIME2025__R10/gpt-oss/p1/results.json

--output_path OUTPUT_PATH
    Path to save output results.json
    Default: output/filtered_reasoning_results.json

--model_path MODEL_PATH (REQUIRED)
    HuggingFace model identifier
    Examples: openai/gpt-oss-120b, meta-llama/Llama-2-7b-hf

--device {cuda, cpu}
    Device for inference
    Default: cuda

--max_reasoning_tokens MAX_REASONING_TOKENS
    Maximum tokens for reasoning generation
    Default: 2000

--limit LIMIT
    Limit number of items to process (useful for testing)
    Default: None (process all items)

--log_level {DEBUG, INFO, WARNING, ERROR}
    Logging verbosity level
    Default: INFO

=============================================================================
PROMPT FORMAT
=============================================================================

Reasoning Stage (Stage 1):
  <|start|>system<|message|>{system_prompt}<|end|>
  <|start|>user<|message|>{question}<|end|>
  <|start|>assistant<|channel|>analysis<|message|>{reasoning_generated}

Answer Stage (Stage 2):
  <|start|>system<|message|>{system_prompt}<|end|>
  <|start|>user<|message|>{question}<|end|>
  <|start|>assistant<|channel|>analysis<|message|>{reasoning}<|end|>
  <|start|>assistant<|channel|>final<|message|>Thus, the answer is{answer_generated}

=============================================================================
PERFORMANCE CONSIDERATIONS
=============================================================================

Memory:
- GPU VRAM: ~24GB for gpt-oss-120b (adjust --max_reasoning_tokens if OOM)
- Token filtering adds minimal overhead (~5% per token)
- Batch size: 1 (sequential processing)

Speed:
- Per-item time: ~30-60 seconds (including 2-stage generation)
- Total time for 300 items: ~3-4 hours on single GPU

Optimization Tips:
- Use --limit for testing before full runs
- Monitor VRAM usage and adjust max_reasoning_tokens if needed
- Use --log_level INFO for progress visibility

=============================================================================
TESTING
=============================================================================

Unit tests included:
  python test_token_filter.py      # Test token filtering logic
  python test_io_format.py         # Test input/output format

=============================================================================
"""

import json
import argparse
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from tqdm import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

from core import (
    build_gpt_oss_prompt_with_reasoning_prefilled_answer,
    parse_answer_from_completion,
)

# =============================================================================
# Logging Setup
# =============================================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# =============================================================================
# Constants
# =============================================================================

MAX_REASONING_TOKENS = 2000
MAX_ANSWER_TOKENS = 200


# =============================================================================
# LogitsProcessor for Token Filtering
# =============================================================================

class LetterOnlyTokenFilter(LogitsProcessor):
    """
    Filter out tokens that consist purely of alphabetic characters.

    This processor identifies tokens like 'hello', 'world', etc. and sets
    their logits to -inf during generation. Mixed tokens like '3x' or 'x_1'
    are allowed.
    """

    def __init__(self, tokenizer):
        """
        Initialize the filter by identifying all purely alphabetic tokens.

        Args:
            tokenizer: HuggingFace tokenizer instance
        """
        self.tokenizer = tokenizer
        self.letter_only_ids = self._identify_letter_only_tokens()
        logger.info(f"Identified {len(self.letter_only_ids)} purely alphabetic tokens")

    def _identify_letter_only_tokens(self) -> set:
        """
        Scan vocabulary and identify tokens that are purely alphabetic.

        Returns:
            Set of token IDs that are purely alphabetic
        """
        letter_only = set()
        for token_id in range(len(self.tokenizer)):
            token_text = self.tokenizer.decode([token_id])
            # Strip whitespace and check if purely alphabetic
            token_stripped = token_text.strip()

            # Only filter if non-empty and purely alphabetic
            if token_stripped and re.match(r'^[A-Za-z]+$', token_stripped):
                letter_only.add(token_id)

        return letter_only

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Apply filtering by setting letter-only token logits to -inf.

        Args:
            input_ids: Token input IDs
            scores: Logits for next token candidates

        Returns:
            Modified scores with letter-only tokens set to -inf
        """
        if self.letter_only_ids:
            scores[:, list(self.letter_only_ids)] = float('-inf')
        return scores


# =============================================================================
# StoppingCriteria for End Token Detection
# =============================================================================

class EndTokenStoppingCriteria(StoppingCriteria):
    """
    Stop generation when <|end|> token is detected.
    """

    def __init__(self, tokenizer, end_token: str = '<|end|>'):
        """
        Initialize the stopping criteria.

        Args:
            tokenizer: HuggingFace tokenizer instance
            end_token: Token string that signals end of reasoning (default: '<|end|>')
        """
        self.tokenizer = tokenizer
        self.end_token = end_token

        # Encode the end token
        self.end_token_ids = tokenizer.encode(end_token, add_special_tokens=False)

        if not self.end_token_ids:
            logger.warning(f"Could not encode end token '{end_token}'")
            self.end_token_ids = []

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
        **kwargs
    ) -> bool:
        """
        Check if end token has been generated.

        Args:
            input_ids: Current token sequence
            scores: Logits (unused)
            **kwargs: Additional arguments

        Returns:
            True if end token detected, False otherwise
        """
        if not self.end_token_ids:
            return False

        if len(self.end_token_ids) == 1:
            # Single token end marker
            return input_ids[0, -1].item() == self.end_token_ids[0]
        else:
            # Multi-token end marker
            last_n_tokens = input_ids[0, -len(self.end_token_ids):]
            expected_ids = torch.tensor(self.end_token_ids, device=input_ids.device)
            return torch.all(last_n_tokens == expected_ids).item()


# =============================================================================
# Model Loading
# =============================================================================

def load_model_and_tokenizer(
    model_path: str,
    device: str = 'cuda'
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer from HuggingFace.

    Args:
        model_path: Model identifier (e.g., 'openai/gpt-oss-120b')
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    logger.info(f"Loading model from {model_path}")

    model_kwargs = {
        "output_attentions": False,
        "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32,
    }

    if device == "cuda":
        model_kwargs["device_map"] = "auto"
        model_kwargs["low_cpu_mem_usage"] = True

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    if device == "cpu":
        model = model.to(device)

    model.eval()

    logger.info(f"Model loaded on device: {model.device}")
    return model, tokenizer


# =============================================================================
# Prompt Building
# =============================================================================

def build_reasoning_prompt(question: str, sys_prompt: str) -> str:
    """
    Build prompt for reasoning generation stage.

    References core.py:build_gpt_oss_prompt_with_reasoning

    Args:
        question: The problem question
        sys_prompt: System prompt

    Returns:
        Complete prompt for reasoning generation
    """
    current_date = datetime.now().strftime("%Y-%m-%d")

    system_message = f"{sys_prompt}\n"
    system_message += "Knowledge cutoff: 2024-06\n"
    system_message += f"Current date: {current_date}\n\n"
    system_message += "Reasoning: high\n\n"
    system_message += "# Valid channels: analysis, commentary, final. Channel must be included for every message."

    prompt = f"<|start|>system<|message|>{system_message}<|end|>"
    prompt += f"<|start|>user<|message|>{question}<|end|>"
    prompt += f"<|start|>assistant<|channel|>analysis<|message|>"

    return prompt


def build_answer_prompt(
    question: str,
    reasoning: str,
    sys_prompt: str
) -> str:
    """
    Build prompt for answer generation stage.

    References core.py:build_gpt_oss_prompt_with_reasoning_prefilled_answer

    Args:
        question: The problem question
        reasoning: Generated reasoning text
        sys_prompt: System prompt

    Returns:
        Complete prompt for answer generation with prefill
    """
    return build_gpt_oss_prompt_with_reasoning_prefilled_answer(
        question=question,
        reasoning=reasoning,
        prefill_text="Thus, the answer is",
        reasoning_effort="high",
        empty_question=False
    )


# =============================================================================
# Generation Functions
# =============================================================================

def generate_reasoning(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    sys_prompt: str,
    max_tokens: int = MAX_REASONING_TOKENS
) -> str:
    """
    Generate reasoning with token filtering.

    Stage 1: Generate reasoning while forbidding purely alphabetic tokens.
    Stop when <|end|> is detected or max_tokens is reached.

    Args:
        model: Loaded model
        tokenizer: Tokenizer
        question: Problem question
        sys_prompt: System prompt
        max_tokens: Maximum tokens for reasoning (default: 2000)

    Returns:
        Generated reasoning text (without <|end|> token)
    """
    # Build prompt
    prompt = build_reasoning_prompt(question, sys_prompt)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs['input_ids']

    logger.debug(f"Reasoning prompt length: {len(input_ids[0])} tokens")

    # Prepare logits processor and stopping criteria
    logits_processor = LogitsProcessorList([
        LetterOnlyTokenFilter(tokenizer)
    ])

    stopping_criteria = StoppingCriteriaList([
        EndTokenStoppingCriteria(tokenizer, '<|end|>')
    ])

    # Generate
    try:
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )

        # Extract generated reasoning (remove prompt part)
        generated_ids = output[0][len(input_ids[0]):]
        reasoning_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        # Clean up <|end|> token if present
        if '<|end|>' in reasoning_text:
            reasoning_text = reasoning_text.split('<|end|>')[0].strip()

        reasoning_text = reasoning_text.strip()

        logger.debug(f"Generated reasoning length: {len(reasoning_text)} chars, {len(generated_ids)} tokens")

        return reasoning_text

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"OOM error during reasoning generation: {e}")
            torch.cuda.empty_cache()
            raise
        else:
            raise


def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    reasoning: str,
    sys_prompt: str,
    max_tokens: int = MAX_ANSWER_TOKENS
) -> str:
    """
    Generate answer based on reasoning.

    Stage 2: Generate answer using prefilled text to guide format.
    No token filtering applied to answer generation.

    Args:
        model: Loaded model
        tokenizer: Tokenizer
        question: Problem question
        reasoning: Generated reasoning text
        sys_prompt: System prompt
        max_tokens: Maximum tokens for answer (default: 200)

    Returns:
        Generated answer text
    """
    # Build prompt with prefill
    prompt = build_answer_prompt(question, reasoning, sys_prompt)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs['input_ids']

    logger.debug(f"Answer prompt length: {len(input_ids[0])} tokens")

    # Generate (no token filtering for answer)
    try:
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.5,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )

        # Extract generated answer
        answer_ids = output[0][len(input_ids[0]):]
        answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)

        # Parse final answer
        final_answer = parse_answer_from_completion(answer_text)

        logger.debug(f"Generated answer: {final_answer}")

        return answer_text

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"OOM error during answer generation: {e}")
            torch.cuda.empty_cache()
            raise
        else:
            raise


# =============================================================================
# Data Processing
# =============================================================================

def process_single_item(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    item: Dict,
    max_reasoning_tokens: int = MAX_REASONING_TOKENS
) -> Dict:
    """
    Process a single data item through both stages.

    Args:
        model: Loaded model
        tokenizer: Tokenizer
        item: Single item from input data
        max_reasoning_tokens: Maximum tokens for reasoning

    Returns:
        Result dictionary with all fields
    """
    unique_id = item['unique_id']
    question = item['question']
    ground_truth = item['answer']
    sys_prompt = item['result']['sys_prompt']

    logger.info(f"Processing: {unique_id}")

    # Stage 1: Generate reasoning
    try:
        reasoning = generate_reasoning(
            model, tokenizer, question, sys_prompt,
            max_tokens=max_reasoning_tokens
        )
    except Exception as e:
        logger.error(f"Failed to generate reasoning for {unique_id}: {e}")
        raise

    # Stage 2: Generate answer
    try:
        answer = generate_answer(
            model, tokenizer, question, reasoning, sys_prompt,
            max_tokens=MAX_ANSWER_TOKENS
        )
    except Exception as e:
        logger.error(f"Failed to generate answer for {unique_id}: {e}")
        raise

    return {
        'unique_id': unique_id,
        'question': question,
        'answer': ground_truth,
        'sys_prompt': sys_prompt,
        'reason_text': reasoning,
        'answer_text': answer
    }


def load_input_data(filepath: str) -> List[Dict]:
    """
    Load input JSON data.

    Args:
        filepath: Path to results.json

    Returns:
        List of data items
    """
    logger.info(f"Loading data from {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} items")
    return data


def save_results(results: List[Dict], output_path: str) -> None:
    """
    Save results to JSON file.

    Args:
        results: List of result dictionaries
        output_path: Path to save output
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving {len(results)} results to {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {output_path}")


# =============================================================================
# Main Program
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Filtered Reasoning Generation Experiment'
    )

    parser.add_argument(
        '--input_path',
        type=str,
        default='data/AIME2025__R10/gpt-oss/p1/results.json',
        help='Path to input results.json'
    )

    parser.add_argument(
        '--output_path',
        type=str,
        default='output/filtered_reasoning_results.json',
        help='Path to save output results'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Model identifier (e.g., openai/gpt-oss-120b)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for inference'
    )

    parser.add_argument(
        '--max_reasoning_tokens',
        type=int,
        default=MAX_REASONING_TOKENS,
        help='Maximum tokens for reasoning generation'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of items to process (for testing)'
    )

    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    args = parser.parse_args()

    # Set logging level
    logger.setLevel(getattr(logging, args.log_level))

    logger.info("=" * 60)
    logger.info("Filtered Reasoning Generation Experiment")
    logger.info("=" * 60)
    logger.info(f"Input:  {args.input_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Model:  {args.model_path}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Max reasoning tokens: {args.max_reasoning_tokens}")

    # Load model
    logger.info("\nLoading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)

    # Load data
    logger.info("\nLoading input data...")
    data = load_input_data(args.input_path)

    if args.limit:
        data = data[:args.limit]
        logger.info(f"Limited to {len(data)} items")

    # Process items
    logger.info(f"\nProcessing {len(data)} items...")
    results = []

    for item in tqdm(data, desc="Processing"):
        try:
            result = process_single_item(
                model, tokenizer, item,
                max_reasoning_tokens=args.max_reasoning_tokens
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {item['unique_id']}: {e}")
            results.append({
                'unique_id': item['unique_id'],
                'question': item['question'],
                'answer': item['answer'],
                'sys_prompt': item['result']['sys_prompt'],
                'reason_text': None,
                'answer_text': None,
                'error': str(e)
            })

    # Save results
    save_results(results, args.output_path)

    # Cleanup
    del model, tokenizer
    if args.device == 'cuda':
        torch.cuda.empty_cache()

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
