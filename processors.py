"""
Processor classes for reasoning text transformations

This module implements the Processor pattern for applying various transformations
to reasoning text:
- MaskProcessor: Mask numbers, answers, or alphabetic characters
- TruncateProcessor: Remove answer lines, last N lines, or percentage of lines
- ShuffleProcessor: Shuffle lines, words, or tokens
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
            mask_alphabet_and_answer_in_reasoning
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
    - 'answer_and_after': Remove answer line and all lines after it
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
        from core import remove_answer_and_after, truncate_reasoning_lines

        self.last_input_stats = self._compute_stats(reasoning)
        input_line_count = self.last_input_stats['lines']

        if self.mode == 'answer_and_after':
            answer = context.get('answer', '')
            result = remove_answer_and_after(reasoning, answer)
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
    - 'token': Shuffle tokens using a tokenizer (kwargs: tokenizer_model)
    """

    def __init__(self, mode: str, seed: int = None, **kwargs):
        """
        Initialize ShuffleProcessor

        Args:
            mode: Shuffle mode (see class docstring)
            seed: Random seed for reproducibility (optional)
            **kwargs: Additional parameters (tokenizer_model for token mode)
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
