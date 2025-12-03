"""
Processor classes for reasoning text transformations

This module implements the Processor pattern for applying various transformations
to reasoning text:
- MaskProcessor: Mask numbers, answers, alphabetic characters, or all non-blank characters
- TruncateProcessor: Remove answer lines, last N lines, or percentage of lines
- ShuffleProcessor: Shuffle lines, words, or tokens
- InsertProcessor: Insert text at random positions
- QuestionProcessor: Modify or remove question text
- RemoveProcessor: Remove or consolidate whitespace characters
- ReplaceProcessor: Replace text using regular expressions
- AnswerProcessor: Add prefill text to guide model answer generation
"""

from abc import ABC, abstractmethod
from typing import Dict


class Processor(ABC):
    """
    Abstract base class for reasoning text processors

    Each processor implements a specific transformation on reasoning text
    and tracks metadata about the transformation.
    """

    @abstractmethod
    def process(self, reasoning: str, context: Dict) -> str:
        """
        Apply the transformation to reasoning text

        Args:
            reasoning: The reasoning text to process
            context: Context dictionary containing question, answer, etc.

        Returns:
            Transformed reasoning text
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict:
        """
        Get metadata about the last processing operation

        Returns:
            Dictionary containing processor type, parameters, and statistics
        """
        pass

    def _compute_stats(self, text: str) -> Dict:
        """
        Compute basic statistics for a text

        Args:
            text: The text to analyze

        Returns:
            Dictionary with lines, chars, and words counts
        """
        lines = [l for l in text.split('\n') if l.strip()]
        words = text.split()
        return {
            'lines': len(lines),
            'chars': len(text),
            'words': len(words)
        }


class MaskProcessor(Processor):
    """
    Processor for masking numbers, answers, or alphabetic characters

    Supported modes:
    - 'number': Mask all numbers
    - 'answer': Mask only the answer
    - 'line': Mask numbers in lines containing the answer
    - 'n-lines': Mask numbers in answer line and N previous lines
    - 'number-advance': Mask computational numbers, preserve algebraic
    - 'alphabet': Mask all alphabetic characters
    - 'alphabet-and-answer': Mask alphabet and answer numbers
    - 'all-nonblank': Mask all non-whitespace characters (letters, numbers, symbols)
                      Preserves spaces, tabs, newlines, and other whitespace characters
    """

    def __init__(self, mode: str, mask_char: str = '█', **kwargs):
        """
        Initialize MaskProcessor

        Args:
            mode: Masking mode (see class docstring)
            mask_char: Character to use for masking (default: '█')
            **kwargs: Additional parameters (e.g., num_prev_lines for 'n-lines' mode)
        """
        self.mode = mode
        self.mask_char = mask_char
        self.kwargs = kwargs
        self.last_input_stats = None
        self.last_output_stats = None

    def process(self, reasoning: str, context: Dict) -> str:
        """Apply masking to reasoning text"""
        from core import (
            mask_numbers_in_reasoning,
            mask_answer_only_in_reasoning,
            mask_numbers_in_lines_with_answer,
            mask_numbers_in_nlines_with_answer,
            mask_numbers_advance,
            mask_alphabet_in_reasoning,
            mask_alphabet_and_answer_in_reasoning,
            mask_all_nonblank_in_reasoning
        )

        self.last_input_stats = self._compute_stats(reasoning)

        answer = context.get('answer', '')

        if self.mode == 'number':
            result = mask_numbers_in_reasoning(reasoning, self.mask_char)
        elif self.mode == 'answer':
            result = mask_answer_only_in_reasoning(reasoning, answer, self.mask_char)
        elif self.mode == 'line':
            result = mask_numbers_in_lines_with_answer(reasoning, answer, self.mask_char)
        elif self.mode == 'n-lines':
            num_prev_lines = self.kwargs.get('num_prev_lines', 1)
            result = mask_numbers_in_nlines_with_answer(
                reasoning, answer, num_prev_lines, self.mask_char
            )
        elif self.mode == 'number-advance':
            result = mask_numbers_advance(reasoning, answer, self.mask_char)
        elif self.mode == 'alphabet':
            result = mask_alphabet_in_reasoning(reasoning, self.mask_char)
        elif self.mode == 'alphabet-and-answer':
            result = mask_alphabet_and_answer_in_reasoning(reasoning, answer, self.mask_char)
        elif self.mode == 'all-nonblank':
            result = mask_all_nonblank_in_reasoning(reasoning, self.mask_char)
        else:
            raise ValueError(f"Invalid mask mode: {self.mode}")

        self.last_output_stats = self._compute_stats(result)
        return result

    def get_metadata(self) -> Dict:
        """Get metadata about the masking operation"""
        metadata = {
            'processor': 'mask',
            'mode': self.mode,
            'mask_char': self.mask_char,
            'input_stats': self.last_input_stats,
            'output_stats': self.last_output_stats
        }
        metadata.update(self.kwargs)
        return metadata


class TruncateProcessor(Processor):
    """
    Processor for truncating reasoning text

    Supported modes:
    - 'all': Remove all reasoning (return empty string)
    - 'answer': Remove exact answer string from reasoning (all occurrences)
    - 'answer_and_after': Remove answer line and all lines after it
    - 'before_answer': Remove all lines before the answer line (answer line kept)
    - 'last_n_lines': Remove last N lines (kwargs: n=5)
    - 'last_ratio': Remove last X% of lines (kwargs: ratio=0.3)
    """

    def __init__(self, mode: str, **kwargs):
        """
        Initialize TruncateProcessor

        Args:
            mode: Truncation mode (see class docstring)
            **kwargs: Additional parameters (n for last_n_lines, ratio for last_ratio)
        """
        self.mode = mode
        self.kwargs = kwargs
        self.last_input_stats = None
        self.last_output_stats = None
        self.removed_lines = 0

    def process(self, reasoning: str, context: Dict) -> str:
        """Apply truncation to reasoning text"""
        from core import (
            remove_answer_and_after,
            remove_before_answer,
            remove_exact_answer,
            truncate_reasoning_lines
        )

        self.last_input_stats = self._compute_stats(reasoning)
        input_line_count = self.last_input_stats['lines']

        if self.mode == 'all':
            result = ''
        elif self.mode == 'answer':
            answer = context.get('answer', '')
            result = remove_exact_answer(reasoning, answer)
        elif self.mode == 'answer_and_after':
            answer = context.get('answer', '')
            result = remove_answer_and_after(reasoning, answer)
        elif self.mode == 'before_answer':
            answer = context.get('answer', '')
            result = remove_before_answer(reasoning, answer)
        elif self.mode == 'last_n_lines':
            n = self.kwargs.get('n', 1)
            result = truncate_reasoning_lines(reasoning, n)
        elif self.mode == 'last_ratio':
            ratio = self.kwargs.get('ratio', 0.1)
            result = truncate_reasoning_lines(reasoning, ratio)
        else:
            raise ValueError(f"Invalid truncate mode: {self.mode}")

        self.last_output_stats = self._compute_stats(result)
        self.removed_lines = input_line_count - self.last_output_stats['lines']

        return result

    def get_metadata(self) -> Dict:
        """Get metadata about the truncation operation"""
        metadata = {
            'processor': 'truncate',
            'mode': self.mode,
            'input_stats': self.last_input_stats,
            'output_stats': self.last_output_stats,
            'removed_lines': self.removed_lines
        }
        metadata.update(self.kwargs)
        return metadata


class ShuffleProcessor(Processor):
    """
    Processor for shuffling reasoning text

    Supported modes:
    - 'line': Shuffle lines
    - 'word': Shuffle words
    - 'token': Shuffle tokens using a tokenizer (kwargs: tokenizer_model or model_type)
    """

    def __init__(self, mode: str, seed: int = 42, **kwargs):
        """
        Initialize ShuffleProcessor

        Args:
            mode: Shuffle mode (see class docstring)
            seed: Random seed for reproducibility (optional)
            **kwargs: Additional parameters
                - tokenizer_model: Explicit tokenizer model name for token mode (optional)
                - model_type: Model type ('deepseek', 'gpt-oss', 'qwen3') to auto-select tokenizer (optional)
        """
        self.mode = mode
        self.seed = seed
        self.kwargs = kwargs
        self.last_input_stats = None
        self.last_output_stats = None

    def process(self, reasoning: str, context: Dict) -> str:
        """Apply shuffling to reasoning text"""
        from core import shuffle_reasoning

        self.last_input_stats = self._compute_stats(reasoning)

        result = shuffle_reasoning(
            reasoning,
            mode=self.mode,
            seed=self.seed,
            **self.kwargs
        )

        self.last_output_stats = self._compute_stats(result)
        return result

    def get_metadata(self) -> Dict:
        """Get metadata about the shuffle operation"""
        metadata = {
            'processor': 'shuffle',
            'mode': self.mode,
            'seed': self.seed,
            'input_stats': self.last_input_stats,
            'output_stats': self.last_output_stats
        }
        metadata.update(self.kwargs)
        return metadata


class InsertProcessor(Processor):
    """
    Processor for inserting text into reasoning

    Supported modes:
    - 'fix': Insert fixed text at random positions

    Position strategy:
    - 'random': Insert at random positions (only supported strategy)

    Count parameter:
    - Integer: Fixed number of insertions (e.g., count=5)
    - Percentage string: Dynamic count based on answer occurrences (e.g., count='100% # of answer')
      Format: '<percentage>% # of answer'
      - '100% # of answer': Insert once for each answer occurrence
      - '200% # of answer': Insert twice for each answer occurrence
      - '50% # of answer': Insert 0.5 times for each answer occurrence (rounded)
    """

    def __init__(self, mode: str, sentence: str = "Maybe the answer is 123.",
                 position: str = 'random', count = 1, seed: int = 42, **kwargs):
        """
        Initialize InsertProcessor

        Args:
            mode: Insertion mode (currently only 'fix' is supported)
            sentence: Text to insert
            position: Position strategy (must be 'random')
            count: Number of times to insert the text (int or percentage string)
            seed: Random seed for reproducibility
            **kwargs: Additional parameters
        """
        self.mode = mode
        self.sentence = sentence
        self.position = position
        self.count = count  # Can be int or str
        self.seed = seed
        self.kwargs = kwargs
        self.last_input_stats = None
        self.last_output_stats = None
        self.insertion_positions = []
        self.original_count = count  # Store original count parameter
        self.actual_count = None  # Will be set during process()

    def process(self, reasoning: str, context: Dict) -> str:
        """Apply insertion to reasoning text"""
        import random

        self.last_input_stats = self._compute_stats(reasoning)

        if self.mode != 'fix':
            raise ValueError(f"Invalid insert mode: {self.mode}. Currently only 'fix' is supported.")

        if self.position != 'random':
            raise ValueError(f"Invalid position strategy: {self.position}. Only 'random' is supported.")

        # Calculate actual count based on count parameter
        actual_count = self._calculate_count(reasoning, context)
        self.actual_count = actual_count

        # Split into non-empty lines
        lines = reasoning.strip().split('\n')
        lines = [line for line in lines if line.strip()]

        self.insertion_positions = []

        # Use seed if provided
        if self.seed is not None:
            random.seed(self.seed)

        # Insert multiple times at random positions
        for _ in range(actual_count):
            # Random position (0 to len(lines), inclusive)
            insert_pos = random.randint(0, len(lines))
            lines.insert(insert_pos, self.sentence)
            self.insertion_positions.append(insert_pos)

        result = '\n'.join(lines)
        self.last_output_stats = self._compute_stats(result)

        return result

    def _calculate_count(self, reasoning: str, context: Dict) -> int:
        """
        Calculate actual insertion count based on count parameter

        Args:
            reasoning: The reasoning text to analyze
            context: Context dictionary containing answer

        Returns:
            Actual number of insertions to perform
        """
        import re

        # If count is an integer, return it directly
        if isinstance(self.count, int):
            return self.count

        # If count is a string, check if it's a percentage format
        if isinstance(self.count, str):
            # Match pattern: "X% # of answer" (case insensitive)
            match = re.match(r'^(\d+(?:\.\d+)?)%\s*#\s*of\s+answer$', self.count.strip(), re.IGNORECASE)
            if match:
                percentage = float(match.group(1))

                # Get answer from context
                answer = context.get('answer') or context.get('ground_truth', '')
                if not answer:
                    raise ValueError(f"Percentage count format '{self.count}' requires 'answer' or 'ground_truth' in context")

                # Count answer occurrences in reasoning (using word boundary)
                pattern = r'\b' + re.escape(str(answer)) + r'\b'
                answer_count = len(re.findall(pattern, reasoning))

                # Calculate actual count: answer_count * (percentage / 100)
                actual_count = answer_count * (percentage / 100.0)

                # Round to nearest integer
                return round(actual_count)
            else:
                raise ValueError(f"Invalid count format: '{self.count}'. Expected integer or 'X% # of answer' format")

        raise ValueError(f"Invalid count type: {type(self.count)}. Expected int or str")

    def get_metadata(self) -> Dict:
        """Get metadata about the insertion operation"""
        metadata = {
            'processor': 'insert',
            'mode': self.mode,
            'sentence': self.sentence,
            'position': self.position,
            'count': self.original_count,  # Original count parameter (may be int or str)
            'actual_count': self.actual_count,  # Computed count (always int)
            'seed': self.seed,
            'insertion_positions': self.insertion_positions,
            'input_stats': self.last_input_stats,
            'output_stats': self.last_output_stats
        }
        metadata.update(self.kwargs)
        return metadata


class QuestionProcessor(Processor):
    """
    Processor for modifying question text

    Supported modes:
    - 'remove': Remove entire question (set to empty string)

    This processor modifies the 'question' field in context instead of the reasoning text.
    """

    def __init__(self, mode: str, **kwargs):
        """
        Initialize QuestionProcessor

        Args:
            mode: Processing mode (currently only 'remove' is supported)
            **kwargs: Additional parameters (reserved for future use)
        """
        self.mode = mode
        self.kwargs = kwargs
        self.original_question = None

    def process(self, reasoning: str, context: Dict) -> str:
        """
        Apply processing to question text in context

        Note: This processor modifies the context['question'] field and returns
        the reasoning text unchanged.
        """
        if self.mode != 'remove':
            raise ValueError(f"Invalid question mode: {self.mode}. Currently only 'remove' is supported.")

        self.original_question = context.get('question', '')

        # Modify context in place
        context['question'] = ''

        # Return reasoning unchanged
        return reasoning

    def get_metadata(self) -> Dict:
        """Get metadata about the question processing operation"""
        metadata = {
            'processor': 'question',
            'mode': self.mode,
            'original_question': self.original_question,
        }
        metadata.update(self.kwargs)
        return metadata


class RemoveProcessor(Processor):
    """
    Processor for removing or consolidating whitespace characters

    Supported modes:
    - 'blank': Consolidate all continuous blank characters (spaces, tabs, newlines)
               into a single space
    """

    def __init__(self, mode: str, **kwargs):
        """
        Initialize RemoveProcessor

        Args:
            mode: Processing mode (currently only 'blank' is supported)
            **kwargs: Additional parameters (reserved for future use)
        """
        self.mode = mode
        self.kwargs = kwargs
        self.last_input_stats = None
        self.last_output_stats = None

    def process(self, reasoning: str, context: Dict) -> str:
        """
        Apply whitespace removal/consolidation to reasoning text

        Args:
            reasoning: The reasoning text to process
            context: Context dictionary (not used in this processor)

        Returns:
            Processed reasoning text with whitespace consolidated
        """
        from core import remove_continuous_blanks

        self.last_input_stats = self._compute_stats(reasoning)

        if self.mode != 'blank':
            raise ValueError(f"Invalid remove mode: {self.mode}. Currently only 'blank' is supported.")

        result = remove_continuous_blanks(reasoning)

        self.last_output_stats = self._compute_stats(result)
        return result

    def get_metadata(self) -> Dict:
        """Get metadata about the removal operation"""
        metadata = {
            'processor': 'remove',
            'mode': self.mode,
            'input_stats': self.last_input_stats,
            'output_stats': self.last_output_stats
        }
        metadata.update(self.kwargs)
        return metadata


class ReplaceProcessor(Processor):
    """
    Processor for replacing text using regular expressions

    This processor allows flexible text replacement using regex patterns.
    Common use cases:
    - Replace all whitespace with single space: pattern=r'\\s', replacement=' '
    - Replace all digits: pattern=r'\\d', replacement='X'
    - Replace specific words or patterns
    - Replace ground truth answer with fixed value: pattern='{ANSWER}', replacement='123'

    Special placeholders:
    - '{ANSWER}' or '{answer}': Replaced with the ground truth answer from context
    """

    def __init__(self, pattern: str, replacement: str = '', **kwargs):
        """
        Initialize ReplaceProcessor

        Args:
            pattern: Regular expression pattern to match, or special placeholder like '{ANSWER}'
            replacement: String to replace matches with (default: empty string)
            **kwargs: Additional parameters (reserved for future use)
        """
        self.pattern = pattern
        self.replacement = replacement
        self.kwargs = kwargs
        self.last_input_stats = None
        self.last_output_stats = None
        self.num_replacements = 0
        self.actual_pattern = None  # Store the actual pattern used after placeholder expansion

    def process(self, reasoning: str, context: Dict) -> str:
        """
        Apply regex replacement to reasoning text

        Args:
            reasoning: The reasoning text to process
            context: Context dictionary containing 'answer' or 'ground_truth' for placeholder expansion

        Returns:
            Processed reasoning text with replacements applied
        """
        import re

        self.last_input_stats = self._compute_stats(reasoning)

        # Expand special placeholders
        actual_pattern = self.pattern
        if self.pattern.upper() == '{ANSWER}':
            # Get ground truth answer from context
            answer = context.get('answer') or context.get('ground_truth', '')
            if not answer:
                raise ValueError("'{ANSWER}' placeholder requires 'answer' or 'ground_truth' in context")
            # Use word boundary to match the answer as a complete word/number
            actual_pattern = r'\b' + re.escape(str(answer)) + r'\b'

        self.actual_pattern = actual_pattern

        # Perform replacement and count occurrences
        result, self.num_replacements = re.subn(actual_pattern, self.replacement, reasoning)

        self.last_output_stats = self._compute_stats(result)
        return result

    def get_metadata(self) -> Dict:
        """Get metadata about the replacement operation"""
        metadata = {
            'processor': 'replace',
            'pattern': self.pattern,
            'actual_pattern': self.actual_pattern,  # Include expanded pattern
            'replacement': self.replacement,
            'num_replacements': self.num_replacements,
            'input_stats': self.last_input_stats,
            'output_stats': self.last_output_stats
        }
        metadata.update(self.kwargs)
        return metadata


class AnswerProcessor(Processor):
    """
    Processor for adding prefill text to guide model answer generation

    This processor does not modify the reasoning text itself.
    Instead, it adds a prefill text to the context that will be used
    when building prompts to guide the model's answer generation.

    Supported modes:
    - 'retrieval': Add prefill text to guide answer generation
    """

    def __init__(self, mode: str, prefill_text: str = "Thus, the answer is", **kwargs):
        """
        Initialize AnswerProcessor

        Args:
            mode: Processing mode (currently only 'retrieval' is supported)
            prefill_text: Text to prefill in the answer section (default: "Thus, the answer is")
            **kwargs: Additional parameters (reserved for future use)
        """
        self.mode = mode
        self.prefill_text = prefill_text
        self.kwargs = kwargs
        self.last_input_stats = None
        self.last_output_stats = None

    def process(self, reasoning: str, context: Dict) -> str:
        """
        Add prefill text to context for answer generation

        This processor does not modify the reasoning text.
        It adds 'answer_prefill' to the context dictionary.

        Args:
            reasoning: The reasoning text (unchanged)
            context: Context dictionary to store prefill text

        Returns:
            Original reasoning text (unchanged)
        """
        self.last_input_stats = self._compute_stats(reasoning)

        if self.mode != 'retrieval':
            raise ValueError(f"Invalid mode: {self.mode}. Currently only 'retrieval' is supported.")

        # Add prefill text to context
        context['answer_prefill'] = self.prefill_text

        self.last_output_stats = self._compute_stats(reasoning)
        return reasoning

    def get_metadata(self) -> Dict:
        """Get metadata about the answer processor operation"""
        metadata = {
            'processor': 'answer',
            'mode': self.mode,
            'prefill_text': self.prefill_text,
            'input_stats': self.last_input_stats,
            'output_stats': self.last_output_stats
        }
        metadata.update(self.kwargs)
        return metadata


class ReasonIsProcessor(Processor):
    """
    Processor for replacing reasoning content with ground truth answer only

    This processor replaces the entire reasoning text with just the answer
    from the context, creating a minimal reasoning chain that only contains
    the correct answer.

    Supported modes:
    - 'answer': Replace reasoning with pure answer from context
    - 'answer_with_illustrate': Replace reasoning with "Thus, the answer is {answer}"
    """

    def __init__(self, mode: str, **kwargs):
        """
        Initialize ReasonIsProcessor

        Args:
            mode: Processing mode (currently only 'answer' is supported)
            **kwargs: Additional parameters (reserved for future use)
        """
        self.mode = mode
        self.kwargs = kwargs
        self.last_input_stats = None
        self.last_output_stats = None

    def process(self, reasoning: str, context: Dict) -> str:
        """
        Replace reasoning text with answer from context

        Args:
            reasoning: The reasoning text (will be replaced)
            context: Context dictionary containing 'answer' or 'ground_truth'

        Returns:
            Answer string (pure or with illustrate format)

        Raises:
            ValueError: If mode is not valid
            KeyError: If 'answer' is not found in context
        """
        self.last_input_stats = self._compute_stats(reasoning)

        if self.mode not in ['answer', 'answer_with_illustrate']:
            raise ValueError(f"Invalid mode for reason_is: {self.mode}. Supported modes: 'answer', 'answer_with_illustrate'")

        # Get answer from context
        if 'answer' not in context:
            raise KeyError("'answer' not found in context")

        answer = context['answer']

        # Format based on mode
        if self.mode == 'answer':
            result = answer
        elif self.mode == 'answer_with_illustrate':
            result = f"Thus, the answer is {answer}"
        else:
            result = answer  # Fallback (should never reach here)

        self.last_output_stats = self._compute_stats(result)
        return result

    def get_metadata(self) -> Dict:
        """Get metadata about the reason_is processor operation"""
        metadata = {
            'processor': 'reason_is',
            'mode': self.mode,
            'input_stats': self.last_input_stats,
            'output_stats': self.last_output_stats
        }
        metadata.update(self.kwargs)
        return metadata
