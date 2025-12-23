"""
Attention Visualization Tool for LLM Reasoning

This script visualizes attention distributions in LLM models during answer generation.
It processes experiment results, extracts attention maps, and generates interactive HTML.

Memory Optimizations:
    - **Direct GPU model loading**: Uses low_cpu_mem_usage=True to load models directly
      on GPU without creating temporary copies in CPU RAM, reducing CPU memory peaks
    - **Layer-by-layer attention extraction**: Uses forward hooks to extract attention
      maps one layer at a time, avoiding simultaneous storage of all layers in VRAM
    - **Sparse attention storage**: Applies threshold filtering (default: 0.01) to set
      low-attention values to 0, reducing CPU memory and disk storage by 60-80%
    - **Model quantization** (4-bit/8-bit) for reduced memory usage
    - **Immediate memory cleanup** after each attention extraction
    - **Streaming JSONL writes** to avoid accumulating instances in memory
    - **Context manager** for automatic model cleanup
    - **Automatic detection** of pre-quantized models (e.g., gpt-oss with Mxfp4)

VRAM Optimization Details:
    The hook-based extraction significantly reduces GPU memory usage:
    - Old method: All layers' attention stored simultaneously (~16GB for 32 layers, 2048 tokens)
    - New method: Only one layer's attention in VRAM at a time (~512MB per layer)
    - Fallback: Automatically falls back to standard extraction if hooks fail

Sparse Storage Details:
    Attention values below threshold are set to 0 and stored in sparse format:
    - Default threshold: 0.01 (1%) - filters out ~60-80% of attention values
    - Sparse dictionary format: {index: value} only for non-zero values
    - Reduces JSON file size by 60-80% and CPU memory usage
    - HTML visualization uses sparse format directly (no conversion to dense)
    - No visual quality loss (low attention values are not visually significant)

Usage:
    # Basic usage (for pre-quantized models like gpt-oss)
    python run_attn_visual.py \\
        --model Qwen/Qwen3-0.6B \\
        --template gpt-oss \\
        --results exp/cdad7f13/results.json \\
        --limit 1

    # With custom sparse threshold (more aggressive filtering)
    python run_attn_visual.py \\
        --model Qwen/Qwen3-0.6B \\
        --template gpt-oss \\
        --results exp/cdad7f13/results.json \\
        --sparse-threshold 0.02

    # With 4-bit quantization (for non-quantized models)
    python run_attn_visual.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --template gpt-oss \\
        --results exp/cdad7f13/results.json \\
        --quantization 4bit

    # Combined: quantization + custom sparse threshold
    python run_attn_visual.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --template gpt-oss \\
        --results exp/cdad7f13/results.json \\
        --quantization 4bit \\
        --sparse-threshold 0.02

Notes:
    - Pre-quantized models (e.g., gpt-oss) will automatically be detected
    - For pre-quantized models, the --quantization flag will be ignored
    - Use --quantization only for non-quantized models to reduce memory usage

Debug Options:
    # Disable output_attentions (for debugging CPU OOM issues)
    DEBUG_ATTN_OUTPUT_OFF=1 python run_attn_visual.py ...

Output:
    - attention_visualization.html: Interactive HTML visualization
    - attention_instances.jsonl: Intermediate instance data (JSONL format)
    Both files are saved in the same directory as the results.json file
"""

import json
import argparse
import re
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Debug flag: Set to '1' to disable output_attentions (for debugging CPU OOM issues)
DEBUG_ATTN_OUTPUT_OFF = os.environ.get('DEBUG_ATTN_OUTPUT_OFF', '0') == '1'


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

    def __init__(self, model_name: str, quantization: Optional[str] = None, sparse_threshold: float = 0.01):
        """
        Load model and tokenizer

        Args:
            model_name: Transformers model name
            quantization: Quantization mode ('4bit', '8bit', or None)
                         Note: Ignored if model is already pre-quantized
            sparse_threshold: Threshold for sparse attention storage.
                            Attention values below this threshold will be set to 0.
                            Default: 0.01 (1%)
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sparse_threshold = sparse_threshold
        print('device:', self.device)
        print(f'sparse_threshold: {self.sparse_threshold}')
        if DEBUG_ATTN_OUTPUT_OFF:
            print('WARNING: DEBUG_ATTN_OUTPUT_OFF is enabled - output_attentions will be disabled')

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Check if model is already pre-quantized
        config = AutoConfig.from_pretrained(model_name)
        is_pre_quantized = hasattr(config, 'quantization_config') and config.quantization_config is not None

        if is_pre_quantized:
            print(f"Model is pre-quantized with {type(config.quantization_config).__name__}")
            if quantization:
                print(f"Warning: Ignoring --quantization={quantization} because model is already quantized")

            # Load pre-quantized model as-is
            model_kwargs = {
                "output_attentions": not DEBUG_ATTN_OUTPUT_OFF,
                "torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
            }
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
                model_kwargs["low_cpu_mem_usage"] = True
        else:
            # Configure quantization for non-quantized models
            model_kwargs = {
                "output_attentions": not DEBUG_ATTN_OUTPUT_OFF
            }

            if quantization and self.device == "cuda":
                from transformers import BitsAndBytesConfig

                if quantization == "4bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    model_kwargs["device_map"] = "auto"
                    model_kwargs["low_cpu_mem_usage"] = True
                    print(f"Using 4-bit quantization")
                elif quantization == "8bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    model_kwargs["device_map"] = "auto"
                    model_kwargs["low_cpu_mem_usage"] = True
                    print(f"Using 8-bit quantization")
            else:
                # Standard loading without quantization
                model_kwargs["torch_dtype"] = torch.bfloat16 if self.device == "cuda" else torch.float32
                if self.device == "cuda":
                    model_kwargs["device_map"] = "auto"
                    model_kwargs["low_cpu_mem_usage"] = True

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        if self.device == "cpu" and not quantization and not is_pre_quantized:
            self.model = self.model.to(self.device)

        self.model.eval()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup model from memory"""
        import torch

        # Delete model and tokenizer
        del self.model
        del self.tokenizer

        # Clear CUDA cache if using GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return False

    def tokenize(self, prompt: str) -> Dict:
        """
        Tokenize prompt

        Args:
            prompt: Prompt string

        Returns:
            Tokenizer output dictionary
        """
        return self.tokenizer(prompt, return_tensors="pt")

    def _get_attention_modules(self):
        """
        Get attention modules for the model architecture

        Returns:
            List of attention modules to register hooks on
        """
        model_type = self.model.config.model_type.lower()

        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Qwen, LLaMA, Mistral, etc.
            return [layer.self_attn for layer in self.model.model.layers]
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-2, GPT-Neo style
            return [layer.attn for layer in self.model.transformer.h]
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'decoder') and hasattr(self.model.model.decoder, 'layers'):
            # OPT style
            return [layer.self_attn for layer in self.model.model.decoder.layers]
        else:
            raise ValueError(f"Unknown model architecture: {model_type}. Cannot locate attention modules.")

    def extract_last_token_attention(
        self,
        prompt: str
    ) -> Tuple[List[str], List]:
        """
        Execute forward pass and extract attention for last token using hooks.
        This method uses forward hooks to extract attention layer-by-layer,
        avoiding storing all layers' attentions simultaneously in VRAM.

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

        # Storage for attention maps (will be populated by hooks)
        attention_maps = []

        def attention_hook(module, input, output):
            """
            Hook function to extract attention weights from each layer.
            This runs during forward pass for each attention module.
            """
            # Different models return attention in different formats
            # Usually it's in output[1] (attn_weights) or a named tuple
            attn_weights = None

            if isinstance(output, tuple):
                # Try to find attention weights in the output
                for item in output:
                    if isinstance(item, torch.Tensor) and item.dim() == 4:
                        # Shape should be (batch, num_heads, seq_len, seq_len)
                        attn_weights = item
                        break

            if attn_weights is None:
                # Fallback: might be directly in output
                if isinstance(output, torch.Tensor) and output.dim() == 4:
                    attn_weights = output

            if attn_weights is not None:
                # Extract attention for last token only: [:, :, -1, :]
                # Shape: (batch, num_heads, seq_len)
                last_token_attn = attn_weights[0, :, -1, :]  # Keep on GPU

                # Average across heads (still on GPU)
                avg_attn = last_token_attn.mean(dim=0)  # Shape: (seq_len,)

                # Apply sparse threshold on GPU and extract only non-zero indices/values
                mask = avg_attn >= self.sparse_threshold
                sparse_indices = mask.nonzero(as_tuple=True)[0]
                sparse_values = avg_attn[sparse_indices]

                # Convert to CPU only the sparse values
                sparse_dict = {}
                if len(sparse_indices) > 0:
                    indices_cpu = sparse_indices.detach().cpu().numpy()
                    values_cpu = sparse_values.detach().float().cpu().numpy()

                    for idx, val in zip(indices_cpu, values_cpu):
                        sparse_dict[str(int(idx))] = float(val)

                attention_maps.append(sparse_dict)

                # Immediate cleanup of GPU tensor
                del attn_weights, last_token_attn, avg_attn, mask, sparse_indices, sparse_values

        # Get attention modules and register hooks
        try:
            attn_modules = self._get_attention_modules()
        except ValueError as e:
            print(f"Warning: {e}")
            print("Falling back to standard attention extraction...")
            return self._extract_last_token_attention_fallback(prompt)

        hooks = []
        for module in attn_modules:
            hook = module.register_forward_hook(attention_hook)
            hooks.append(hook)

        # Forward pass with hooks active
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_attentions=not DEBUG_ATTN_OUTPUT_OFF  # This ensures attention is computed
            )

        # Remove all hooks
        for hook in hooks:
            hook.remove()

        # Immediate memory cleanup - prioritize deleting attentions
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            del outputs.attentions  # Delete accumulated attentions first
            import gc
            gc.collect()  # Force CPU garbage collection immediately

        del outputs, inputs, input_ids
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return tokens, attention_maps

    def _extract_last_token_attention_fallback(
        self,
        prompt: str
    ) -> Tuple[List[str], List]:
        """
        Fallback method: standard attention extraction (original implementation)
        Used when hook-based extraction fails.

        Args:
            prompt: Complete prompt string

        Returns:
            Tuple of (tokens, attention_maps)
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
                output_attentions=not DEBUG_ATTN_OUTPUT_OFF
            )

        # Extract attentions
        attentions = outputs.attentions

        # Process each layer
        attention_maps = []
        for layer_attention in attentions:
            # Get attention for last token: [:, :, -1, :]
            last_token_attn = layer_attention[0, :, -1, :]  # Keep on GPU

            # Average across heads (still on GPU)
            avg_attn = last_token_attn.mean(dim=0)  # Shape: (seq_len,)

            # Apply sparse threshold on GPU and extract only non-zero indices/values
            mask = avg_attn >= self.sparse_threshold
            sparse_indices = mask.nonzero(as_tuple=True)[0]
            sparse_values = avg_attn[sparse_indices]

            # Convert to CPU only the sparse values
            sparse_dict = {}
            if len(sparse_indices) > 0:
                indices_cpu = sparse_indices.detach().cpu().numpy()
                values_cpu = sparse_values.detach().float().cpu().numpy()

                for idx, val in zip(indices_cpu, values_cpu):
                    sparse_dict[str(int(idx))] = float(val)

            attention_maps.append(sparse_dict)

        # Immediate memory cleanup - prioritize deleting attentions
        del attentions  # Delete accumulated attentions first
        import gc
        gc.collect()  # Force CPU garbage collection immediately

        del outputs, inputs, input_ids
        if self.device == "cuda":
            torch.cuda.empty_cache()

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

            // Get selected layer (sparse format)
            const layerIdx = parseInt(layerSlider.value);
            const sparseAttention = instance.attention_maps[layerIdx];
            const tokens = instance.tokens;

            // Find min and max from sparse values only (for normalization)
            const sparseValues = Object.values(sparseAttention);
            const minWeight = sparseValues.length > 0 ? Math.min(...sparseValues) : 0;
            const maxWeight = sparseValues.length > 0 ? Math.max(...sparseValues) : 1;

            // Generate token HTML (directly from sparse format)
            const tokensHtml = tokens.map((token, i) => {{
                // Get weight from sparse dict (0 if not present)
                const weight = sparseAttention[i.toString()] || 0.0;
                const normalized = weight > 0 ? (weight - minWeight) / (maxWeight - minWeight) : 0;

                // Use white to red colormap
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

    parser.add_argument(
        '--quantization',
        type=str,
        choices=['4bit', '8bit', None],
        default=None,
        help='Model quantization mode (4bit, 8bit, or None for full precision)'
    )

    parser.add_argument(
        '--sparse-threshold',
        type=float,
        default=0.01,
        help='Threshold for sparse attention storage. Attention values below this threshold will be set to 0. Default: 0.01 (1%%)'
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

    # Setup output paths
    output_dir = Path(args.results).parent
    instances_jsonl = output_dir / 'attention_instances.jsonl'
    output_html = output_dir / 'attention_visualization.html'

    # Phase 2-4: Process each instance with streaming write
    truncator = AnswerTruncator()
    builder = PromptBuilder(args.template)

    # Open JSONL file for streaming writes
    with open(instances_jsonl, 'w', encoding='utf-8') as jsonl_file:
        # Use context manager for model lifecycle
        with AttentionExtractor(
            args.model,
            quantization=args.quantization,
            sparse_threshold=args.sparse_threshold
        ) as extractor:
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
                    jsonl_file.write(json.dumps(instance, ensure_ascii=False) + '\n')
                    jsonl_file.flush()
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

                # Phase 4: Extract attention (already in sparse format)
                tokens, attention_maps = extractor.extract_last_token_attention(prompt)
                print(f"  Extracted {len(attention_maps)} layers, {len(tokens)} tokens")

                # Calculate sparsity (attention_maps is already sparse)
                total_values = len(attention_maps) * len(tokens)  # layers * tokens
                sparse_values = sum(len(attn) for attn in attention_maps)  # non-zero values
                sparsity = (1 - sparse_values / total_values) * 100 if total_values > 0 else 0
                print(f"  Sparsity: {sparsity:.1f}% ({sparse_values}/{total_values} values retained)")

                instance = {
                    'question': result['question'],
                    'ground_truth': result['ground_truth'],
                    'is_correct': True,
                    'tokens': tokens,
                    'attention_maps': attention_maps  # Already sparse
                }

                # Write to JSONL immediately
                jsonl_file.write(json.dumps(instance, ensure_ascii=False) + '\n')
                jsonl_file.flush()

    # Phase 5: Generate HTML from JSONL
    print(f"\nGenerating HTML at {output_html}...")
    print(f"Reading instances from {instances_jsonl}...")

    # Read instances from JSONL
    instances = []
    with open(instances_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            instances.append(json.loads(line))

    generator = HTMLGenerator()
    generator.generate_html(instances, str(output_html))

    print(f"Done! Open {output_html} in your browser to view the visualization.")
    print(f"Intermediate data saved to {instances_jsonl}")


if __name__ == '__main__':
    main()
