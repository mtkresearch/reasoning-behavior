"""
Attention Visualization Tool for LLM Reasoning

This script visualizes attention distributions in LLM models during answer generation.
It processes experiment results, extracts attention maps, and generates interactive HTML.

Usage:
    python run_attn_visual.py \\
        --model Qwen/Qwen3-0.6B \\
        --template gpt-oss \\
        --results exp/cdad7f13/results.json \\
        --limit 1

Output:
    Generates attention_visualization.html in the same directory as the results.json file
"""

import json
import argparse
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path


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

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
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
        import numpy as np

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
            last_token_attn = layer_attention[0, :, -1, :].cpu().numpy()

            # Average across heads
            avg_attn = last_token_attn.mean(axis=0)

            attention_maps.append(avg_attn)

        return tokens, attention_maps


# =============================================================================
# Phase 5: HTMLGenerator
# =============================================================================

class HTMLGenerator:
    """Generate interactive HTML visualization"""

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
        html_content = self._build_html_structure(instances)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _build_html_structure(self, instances: List[Dict]) -> str:
        """Build complete HTML structure"""
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Attention Visualization</title>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    <h1>Attention Visualization</h1>

    <div class="controls">
        <label>Instance:</label>
        <select id="instance-select">
            {self._build_instance_options(instances)}
        </select>

        <div class="layer-control">
            <label>Layer:</label>
            <span id="layer-value">Layer 0</span>
            <div class="slider-container">
                <input type="range" id="layer-slider" min="0" max="0" value="0" step="1" />
                <div id="slider-ticks"></div>
            </div>
        </div>
    </div>

    <div id="metadata">
        <p><strong>Question:</strong> <span id="question"></span></p>
        <p><strong>Ground Truth:</strong> <span id="ground-truth"></span></p>
        <p><strong>Correct:</strong> <span id="is-correct"></span></p>
    </div>

    <div id="tokens-container">
        <!-- Tokens will be populated by JavaScript -->
    </div>

    <script>
        {self._get_javascript(instances)}
    </script>
</body>
</html>"""

    def _get_css(self) -> str:
        """Get CSS styles"""
        return """
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }

        h1 {
            color: #333;
        }

        .controls {
            margin: 20px 0;
            padding: 15px;
            padding-bottom: 35px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .controls label {
            margin-right: 10px;
            font-weight: bold;
        }

        .controls select {
            margin-right: 20px;
            padding: 5px 10px;
            font-size: 14px;
            border: 1px solid #ddd;
            border-radius: 3px;
        }

        .layer-control {
            display: inline-block;
            margin-left: 20px;
        }

        .slider-container {
            position: relative;
            width: 900px;
            margin-top: 10px;
            height: 45px;
        }

        #layer-slider {
            width: 100%;
            margin: 0;
            position: relative;
            z-index: 2;
        }

        #layer-value {
            font-weight: bold;
            color: #333;
            margin-left: 10px;
            font-size: 16px;
        }

        #slider-ticks {
            position: absolute;
            top: 25px;
            left: 0;
            width: 100%;
            height: 30px;
            display: flex;
            justify-content: space-between;
            pointer-events: none;
        }

        .tick {
            position: relative;
            width: 2px;
            height: 8px;
            background-color: #666;
        }

        .tick-label {
            position: absolute;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 11px;
            color: #666;
            white-space: nowrap;
        }

        #metadata {
            margin: 20px 0;
            padding: 15px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        #metadata p {
            margin: 8px 0;
        }

        #tokens-container {
            margin: 20px 0;
            padding: 15px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            line-height: 2;
        }

        .token {
            display: inline-block;
            padding: 2px 4px;
            margin: 1px;
            border-radius: 2px;
            font-family: monospace;
            font-size: 7px;
            white-space: pre-wrap;
            word-break: break-word;
        }

        .incorrect-message {
            color: #d32f2f;
            font-size: 18px;
            font-weight: bold;
            padding: 20px;
            text-align: center;
        }
        """

    def _build_instance_options(self, instances: List[Dict]) -> str:
        """Build HTML options for instance select"""
        options = []
        for i, instance in enumerate(instances):
            status = "CORRECT" if instance['is_correct'] else "INCORRECT"
            options.append(
                f'<option value="{i}">Instance {i} ({status})</option>'
            )
        return '\n'.join(options)

    def _get_javascript(self, instances: List[Dict]) -> str:
        """Get JavaScript code"""
        import json

        # Convert instances to JSON
        instances_json = json.dumps(instances, ensure_ascii=False)

        return f"""
        const instances = {instances_json};

        function updateVisualization() {{
            const instanceIdx = parseInt(document.getElementById('instance-select').value);
            const instance = instances[instanceIdx];

            // Update metadata
            document.getElementById('question').textContent = instance.question;
            document.getElementById('ground-truth').textContent = instance.ground_truth;
            document.getElementById('is-correct').textContent = instance.is_correct ? 'Yes' : 'No';

            // Update layer slider if needed
            const layerSlider = document.getElementById('layer-slider');
            if (instance.is_correct && instance.attention_maps.length > 0) {{
                const numLayers = instance.attention_maps.length;
                layerSlider.max = numLayers - 1;
                layerSlider.value = Math.min(parseInt(layerSlider.value), numLayers - 1);
                document.getElementById('layer-value').textContent = `Layer ${{layerSlider.value}}`;

                // Update slider ticks
                updateSliderTicks(numLayers);
            }}

            // Check if correct
            if (!instance.is_correct) {{
                document.getElementById('tokens-container').innerHTML =
                    '<div class="incorrect-message">INCORRECT - No visualization available</div>';
                return;
            }}

            // Get selected layer
            const layerIdx = parseInt(layerSlider.value);
            const attentionWeights = instance.attention_maps[layerIdx];
            const tokens = instance.tokens;

            // Find min and max for normalization
            const minWeight = Math.min(...attentionWeights);
            const maxWeight = Math.max(...attentionWeights);

            // Generate token HTML
            const tokensHtml = tokens.map((token, i) => {{
                const weight = attentionWeights[i];
                const normalized = (weight - minWeight) / (maxWeight - minWeight);

                // Use viridis colormap (simplified)
                const color = getColor(normalized);

                return `<span class="token" style="background-color: ${{color}}" title="Weight: ${{weight.toFixed(4)}}">${{escapeHtml(token)}}</span>`;
            }}).join('');

            document.getElementById('tokens-container').innerHTML = tokensHtml;
        }}

        function updateSliderTicks(numLayers) {{
            const ticksContainer = document.getElementById('slider-ticks');
            ticksContainer.innerHTML = '';

            // Create tick marks for each layer
            for (let i = 0; i < numLayers; i++) {{
                const tick = document.createElement('div');
                tick.className = 'tick';

                const label = document.createElement('div');
                label.className = 'tick-label';
                label.textContent = i;

                tick.appendChild(label);
                ticksContainer.appendChild(tick);
            }}
        }}

        function getColor(value) {{
            // White to Red colormap
            // value in [0, 1], 0 = white (255,255,255), 1 = red (255,0,0)
            const r = 255;
            const g = Math.floor(255 * (1 - value));
            const b = Math.floor(255 * (1 - value));
            return `rgb(${{r}}, ${{g}}, ${{b}})`;
        }}

        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}

        // Event listeners
        document.getElementById('instance-select').addEventListener('change', () => {{
            // Reset layer slider when instance changes
            const layerSlider = document.getElementById('layer-slider');
            layerSlider.value = 0;
            updateVisualization();
        }});

        document.getElementById('layer-slider').addEventListener('input', () => {{
            const layerIdx = document.getElementById('layer-slider').value;
            document.getElementById('layer-value').textContent = `Layer ${{layerIdx}}`;
            updateVisualization();
        }});

        // Keyboard navigation
        document.addEventListener('keydown', (event) => {{
            const instanceSelect = document.getElementById('instance-select');
            const layerSlider = document.getElementById('layer-slider');

            switch(event.key) {{
                case 'ArrowLeft':
                    // Decrease layer
                    event.preventDefault();
                    if (parseInt(layerSlider.value) > parseInt(layerSlider.min)) {{
                        layerSlider.value = parseInt(layerSlider.value) - 1;
                        document.getElementById('layer-value').textContent = `Layer ${{layerSlider.value}}`;
                        updateVisualization();
                    }}
                    break;

                case 'ArrowRight':
                    // Increase layer
                    event.preventDefault();
                    if (parseInt(layerSlider.value) < parseInt(layerSlider.max)) {{
                        layerSlider.value = parseInt(layerSlider.value) + 1;
                        document.getElementById('layer-value').textContent = `Layer ${{layerSlider.value}}`;
                        updateVisualization();
                    }}
                    break;

                case 'ArrowUp':
                    // Previous instance
                    event.preventDefault();
                    if (instanceSelect.selectedIndex > 0) {{
                        instanceSelect.selectedIndex--;
                        layerSlider.value = 0;
                        updateVisualization();
                    }}
                    break;

                case 'ArrowDown':
                    // Next instance
                    event.preventDefault();
                    if (instanceSelect.selectedIndex < instanceSelect.options.length - 1) {{
                        instanceSelect.selectedIndex++;
                        layerSlider.value = 0;
                        updateVisualization();
                    }}
                    break;
            }}
        }});

        // Initial render
        updateVisualization();
        """


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

    # Phase 2-5: Process each instance
    instances = []
    truncator = AnswerTruncator()
    builder = PromptBuilder(args.template)
    extractor = AttentionExtractor(args.model)

    for i, result in enumerate(valid_results):
        print(f"\nProcessing instance {i}...")

        # Check correctness
        if not result['is_correct']:
            print(f"  Skipping incorrect result")
            instances.append({
                'question': result['question'],
                'ground_truth': result['ground_truth'],
                'is_correct': False,
                'tokens': [],
                'attention_maps': []
            })
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

        instances.append({
            'question': result['question'],
            'ground_truth': result['ground_truth'],
            'is_correct': True,
            'tokens': tokens,
            'attention_maps': [attn.tolist() for attn in attention_maps]
        })

    # Phase 5: Generate HTML
    output_path = Path(args.results).parent / 'attention_visualization.html'
    print(f"\nGenerating HTML at {output_path}...")

    generator = HTMLGenerator()
    generator.generate_html(instances, str(output_path))

    print(f"Done! Open {output_path} in your browser to view the visualization.")


if __name__ == '__main__':
    main()
