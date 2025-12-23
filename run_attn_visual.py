"""
Attention Visualization Tool for LLM Reasoning

This script visualizes attention distributions in LLM models during answer generation.
It processes experiment results, extracts attention maps, and generates interactive HTML.

Features:
    - Streaming HTML generation to avoid memory accumulation
    - Automatic memory cleanup after each instance
    - GPU memory management with torch.cuda.empty_cache()
    - Interactive visualization with layer-by-layer attention viewing

Usage:
    python run_attn_visual.py \\
        --model Qwen/Qwen3-0.6B \\
        --template gpt-oss \\
        --results exp/cdad7f13/results.json \\
        --limit 1

Output:
    Generates attention_visualization.html in the same directory as the results.json file

Memory Optimization:
    - Uses HTMLStreamWriter to write instances incrementally to HTML
    - Cleans up GPU/CPU memory after processing each instance
    - No instance accumulation in memory
"""

import json
import argparse
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from attention_visual_templates import (
    get_html_header,
    get_html_footer,
    get_javascript_code,
    build_complete_html
)


# =============================================================================
# Phase 1: DataLoader
# =============================================================================

class DataLoader:
    """Load and validate experiment results data"""

    def load_results(self, path: str) -> List[Dict]:
        """
        Load results.json file

        Args:
            path: Path to results.json file

        Returns:
            List of result dictionaries

        Raises:
            FileNotFoundError: If file does not exist
        """
        with open(path, 'r') as f:
            data = json.load(f)

        # Handle both flat list and nested 'results' key
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'results' in data:
            return data['results']
        else:
            raise ValueError("Invalid JSON format: expected list or dict with 'results' key")

    def validate_result(self, result: Dict) -> bool:
        """
        Validate that result contains all required fields

        Args:
            result: Result dictionary

        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            'question',
            'ground_truth',
            'processed_reasoning',
            'generated_answer',
            'is_correct'
        ]

        return all(field in result for field in required_fields)

    def filter_correct_only(self, results: List[Dict]) -> List[Dict]:
        """
        Filter results to only include is_correct=True

        Args:
            results: List of result dictionaries

        Returns:
            Filtered list containing only correct results
        """
        return [r for r in results if r.get('is_correct', False)]


# =============================================================================
# Phase 2: AnswerTruncator
# =============================================================================

class AnswerTruncator:
    """Handle answer truncation logic"""

    def find_answer_position(self, text: str, ground_truth: str) -> Optional[int]:
        """
        Find the first position where ground_truth appears in text

        Args:
            text: Text to search in
            ground_truth: Answer to find

        Returns:
            Position of first match, or None if not found
        """
        # Handle empty ground truth
        if not ground_truth:
            return None

        # Escape special regex characters
        escaped_pattern = re.escape(ground_truth)

        # For alphanumeric answers: use word boundaries (strict matching)
        if ground_truth[0].isalnum() and ground_truth[-1].isalnum():
            pattern = r'\b' + escaped_pattern + r'\b'
            match = re.search(pattern, text)
            return match.start() if match else None

        # For special characters: direct match without word boundaries
        match = re.search(escaped_pattern, text)
        return match.start() if match else None

    def truncate_at_answer(self, text: str, position: int) -> str:
        """
        Truncate text at the specified position

        Args:
            text: Text to truncate
            position: Position to truncate at

        Returns:
            Truncated text (everything before position)
        """
        return text[:position]

    def process(self, generated_answer: str, ground_truth: str) -> str:
        """
        Execute complete truncation workflow

        Args:
            generated_answer: Generated answer text
            ground_truth: Correct answer

        Returns:
            Truncated answer text
        """
        position = self.find_answer_position(generated_answer, ground_truth)

        if position is None:
            # If answer not found, return full text
            return generated_answer

        return self.truncate_at_answer(generated_answer, position)


# =============================================================================
# Phase 3: PromptBuilder
# =============================================================================

class PromptBuilder:
    """Assemble complete prompt using specified template"""

    def __init__(self, template: str):
        """
        Initialize with template type

        Args:
            template: Template type (e.g., 'gpt-oss')
        """
        self.template = template

    def build_prompt(
        self,
        question: str,
        reasoning: str,
        prefill_text: str,
        truncated_answer: str
    ) -> str:
        """
        Assemble complete prompt

        Args:
            question: Question text
            reasoning: Processed reasoning
            prefill_text: Answer prefix (e.g., "Thus, the answer is")
            truncated_answer: Truncated answer text

        Returns:
            Complete prompt string
        """
        if self.template == 'gpt-oss':
            from core import build_gpt_oss_prompt_with_reasoning_prefilled_answer

            # Merge prefill_text and truncated_answer
            full_answer = f"{prefill_text} {truncated_answer}".strip()

            # Build prompt with merged answer
            return build_gpt_oss_prompt_with_reasoning_prefilled_answer(
                question=question,
                reasoning=reasoning,
                prefill_text=full_answer
            )
        else:
            raise ValueError(f"Unknown template: {self.template}")


# =============================================================================
# Phase 4: AttentionExtractor
# =============================================================================

class AttentionExtractor:
    """Extract and process attention maps from model"""

    def __init__(self, model_name: str):
        """
        Load model and tokenizer

        Args:
            model_name: Transformers model name
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print('device:', self.device)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            output_attentions=True
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()

    def tokenize(self, prompt: str) -> Dict:
        """
        Tokenize prompt

        Args:
            prompt: Prompt string

        Returns:
            Tokenizer output dictionary
        """
        return self.tokenizer(prompt, return_tensors="pt")

    def extract_last_token_attention(
        self,
        prompt: str
    ) -> Tuple[List[str], List]:
        """
        Execute forward pass and extract attention for last token

        Args:
            prompt: Complete prompt string

        Returns:
            Tuple of (tokens, attention_maps) where:
                - tokens: List of token strings
                - attention_maps: List of numpy arrays (num_layers, seq_len)
                  Each array represents averaged attention across all heads
        """
        import torch
        import gc

        # Tokenize
        inputs = self.tokenize(prompt)
        input_ids = inputs['input_ids'].to(self.device)

        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_attentions=True
            )

        # Extract attentions
        # outputs.attentions: tuple of (batch, num_heads, seq_len, seq_len) for each layer
        attentions = outputs.attentions

        # Process each layer
        attention_maps = []
        for layer_attention in attentions:
            # layer_attention: (batch, num_heads, seq_len, seq_len)
            # Get attention for last token: [:, :, -1, :]
            last_token_attn = layer_attention[0, :, -1, :].float().cpu().numpy()

            # Average across heads
            avg_attn = last_token_attn.mean(axis=0)

            attention_maps.append(avg_attn)

        # Memory cleanup
        del inputs, input_ids, outputs, attentions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return tokens, attention_maps


# =============================================================================
# Phase 5: HTMLGenerator
# =============================================================================

class HTMLStreamWriter:
    """Stream HTML generation to avoid memory accumulation"""

    def __init__(self, output_path: str):
        """
        Initialize stream writer

        Args:
            output_path: Path to save HTML file
        """
        self.output_path = output_path
        self.file_handle = None
        self.instance_count = 0

    def __enter__(self):
        """Context manager entry"""
        self.file_handle = open(self.output_path, 'w', encoding='utf-8')
        self._write_html_header()
        self._write_javascript_start()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit"""
        if self.file_handle:
            self._write_javascript_end()
            self._write_html_footer()
            self.file_handle.close()

    def add_instance(self, instance: Dict) -> None:
        """
        Add a single instance to the HTML stream

        Args:
            instance: Data point containing:
                - question: Question text
                - ground_truth: Ground truth answer
                - is_correct: Whether answer is correct
                - tokens: List of token strings
                - attention_maps: List of attention arrays (num_layers, seq_len)
        """
        import json

        # Add comma separator if not first instance
        if self.instance_count > 0:
            self.file_handle.write(',\n')

        # Write instance as JSON
        json_str = json.dumps(instance, ensure_ascii=False, indent=2)
        self.file_handle.write(json_str)
        self.file_handle.flush()

        self.instance_count += 1

    def _write_html_header(self) -> None:
        """Write HTML header and controls"""
        header = get_html_header()
        self.file_handle.write(header)
        self.file_handle.flush()

    def _write_javascript_start(self) -> None:
        """Write JavaScript array start"""
        self.file_handle.write('        const instances = [\n')
        self.file_handle.flush()

    def _write_javascript_end(self) -> None:
        """Write JavaScript functions and event listeners"""
        js_code = f"""
        ];

        {get_javascript_code()}
"""
        self.file_handle.write(js_code)
        self.file_handle.flush()

    def _write_html_footer(self) -> None:
        """Write HTML footer"""
        footer = get_html_footer()
        self.file_handle.write(footer)
        self.file_handle.flush()


class HTMLGenerator:
    """Generate interactive HTML visualization (legacy batch mode)"""

    def generate_html(
        self,
        instances: List[Dict],
        output_path: str
    ) -> None:
        """
        Generate complete HTML file

        Args:
            instances: List of data points, each containing:
                - question: Question text
                - ground_truth: Ground truth answer
                - is_correct: Whether answer is correct
                - tokens: List of token strings
                - attention_maps: List of attention arrays (num_layers, seq_len)
            output_path: Path to save HTML file
        """
        import json

        # Build instance options
        instance_options = self._build_instance_options(instances)

        # Convert instances to JSON
        instances_json = json.dumps(instances, ensure_ascii=False)

        # Build complete HTML using external template
        html_content = build_complete_html(instances_json, instance_options)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _build_instance_options(self, instances: List[Dict]) -> str:
        """Build HTML options for instance select"""
        options = []
        for i, instance in enumerate(instances):
            status = "CORRECT" if instance['is_correct'] else "INCORRECT"
            options.append(
                f'<option value="{i}">Instance {i} ({status})</option>'
            )
        return '\n'.join(options)


# =============================================================================
# Main Program
# =============================================================================

def main():
    """Main program"""
    parser = argparse.ArgumentParser(description='Visualize attention distributions in LLM reasoning')

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Transformers model name (e.g., Qwen/Qwen3-0.6B)'
    )

    parser.add_argument(
        '--template',
        type=str,
        required=True,
        help='Prompt template type (e.g., gpt-oss)'
    )

    parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='Path to results.json file'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of data points to process'
    )

    args = parser.parse_args()

    # Phase 1: Load data
    print("Loading results...")
    loader = DataLoader()
    results = loader.load_results(args.results)

    # Validate and filter
    valid_results = [r for r in results if loader.validate_result(r)]
    print(f"Loaded {len(valid_results)} valid results")

    # Apply limit
    if args.limit:
        valid_results = valid_results[:args.limit]
        print(f"Limited to {len(valid_results)} results")

    # Phase 2-5: Process each instance with streaming HTML generation
    truncator = AnswerTruncator()
    builder = PromptBuilder(args.template)
    extractor = AttentionExtractor(args.model)

    # Initialize streaming HTML writer
    output_path = Path(args.results).parent / 'attention_visualization.html'
    print(f"\nGenerating HTML at {output_path}...")

    with HTMLStreamWriter(str(output_path)) as html_writer:
        for i, result in enumerate(valid_results):
            print(f"\nProcessing instance {i}...")

            # Check correctness
            if not result['is_correct']:
                print(f"  Skipping incorrect result")
                instance = {
                    'question': result['question'],
                    'ground_truth': result['ground_truth'],
                    'is_correct': False,
                    'tokens': [],
                    'attention_maps': []
                }
                html_writer.add_instance(instance)
                continue

            # Phase 2: Truncate answer
            truncated = truncator.process(
                result['generated_answer'],
                result['ground_truth']
            )
            print(f"  Truncated answer length: {len(truncated)}")

            # Phase 3: Build prompt
            prompt = builder.build_prompt(
                question=result['question'],
                reasoning=result['processed_reasoning'],
                prefill_text="Thus, the answer is",
                truncated_answer=truncated
            )
            print(f"  Built prompt length: {len(prompt)}")

            # Phase 4: Extract attention
            tokens, attention_maps = extractor.extract_last_token_attention(prompt)
            print(f"  Extracted {len(attention_maps)} layers, {len(tokens)} tokens")

            # Create instance and stream to HTML
            instance = {
                'question': result['question'],
                'ground_truth': result['ground_truth'],
                'is_correct': True,
                'tokens': tokens,
                'attention_maps': [attn.tolist() for attn in attention_maps]
            }
            html_writer.add_instance(instance)

            # Memory cleanup
            import gc
            import torch
            del instance, tokens, attention_maps, prompt, truncated
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            print(f"  Written to HTML and cleaned up memory")

    print(f"\nDone! Open {output_path} in your browser to view the visualization.")


if __name__ == '__main__':
    main()
