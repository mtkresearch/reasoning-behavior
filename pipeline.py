"""
Pipeline management and flow parsing

This module implements:
- Pipeline: Manages sequential execution of processors
- parse_flow: Parses flow strings into processor lists
"""

import re
from typing import List, Tuple, Dict
from processors import Processor, MaskProcessor, TruncateProcessor, ShuffleProcessor, InsertProcessor, QuestionProcessor, RemoveProcessor, ReplaceProcessor, AnswerProcessor, ReasonIsProcessor, RandomProcessor


class Pipeline:
    """
    Pipeline for executing a sequence of processors

    The pipeline executes processors in order, passing the output of one
    processor as the input to the next. It also collects metadata from
    each processor.
    """

    def __init__(self, processors: List[Processor]):
        """
        Initialize pipeline with a list of processors

        Args:
            processors: List of Processor instances to execute in sequence
        """
        self.processors = processors

    def execute(self, reasoning: str, context: Dict) -> Tuple[str, List[Dict]]:
        """
        Execute the complete pipeline

        Args:
            reasoning: Input reasoning text
            context: Context dictionary (question, answer, etc.)

        Returns:
            Tuple of (processed_reasoning, metadata_list)
        """
        result = reasoning
        metadata_list = []

        for processor in self.processors:
            result = processor.process(result, context)
            metadata_list.append(processor.get_metadata())

        return result, metadata_list


def parse_flow(flow_str: str) -> List[Processor]:
    """
    Parse flow string into list of processors

    Flow syntax:
        processor_type('mode', param1=value1, param2=value2)

    Examples:
        - "mask('number')"
        - "mask('number',mask_char='*')"
        - "truncate('last_ratio',ratio=0.3)"
        - "shuffle('line',seed=42)"
        - "mask('number'),shuffle('line')"

    Args:
        flow_str: Flow string to parse

    Returns:
        List of Processor instances

    Raises:
        ValueError: If processor type is unknown or syntax is invalid
    """
    processors = []

    # Split by comma (but not within quotes or parentheses)
    steps = _split_flow_steps(flow_str)

    for step in steps:
        step = step.strip()
        if not step:
            continue

        processor = _parse_step(step)
        processors.append(processor)

    return processors


def _split_flow_steps(flow_str: str) -> List[str]:
    """
    Split flow string by commas, respecting parentheses and quotes

    Args:
        flow_str: Flow string to split

    Returns:
        List of step strings
    """
    steps = []
    current_step = []
    paren_depth = 0
    in_quote = False
    quote_char = None

    for char in flow_str:
        if char in ('"', "'") and not in_quote:
            in_quote = True
            quote_char = char
            current_step.append(char)
        elif char == quote_char and in_quote:
            in_quote = False
            quote_char = None
            current_step.append(char)
        elif char == '(' and not in_quote:
            paren_depth += 1
            current_step.append(char)
        elif char == ')' and not in_quote:
            paren_depth -= 1
            current_step.append(char)
        elif char == ',' and paren_depth == 0 and not in_quote:
            # This is a step separator
            steps.append(''.join(current_step))
            current_step = []
        else:
            current_step.append(char)

    # Add last step
    if current_step:
        steps.append(''.join(current_step))

    return steps


def _parse_step(step: str) -> Processor:
    """
    Parse a single step into a Processor instance

    Args:
        step: Step string (e.g., "mask('number',mask_char='*')")

    Returns:
        Processor instance

    Raises:
        ValueError: If processor type is unknown or syntax is invalid
    """
    # Match: processor_type(arguments)
    match = re.match(r'(\w+)\((.*)\)', step.strip())
    if not match:
        raise ValueError(f"Invalid step syntax: {step}")

    processor_type = match.group(1)
    args_str = match.group(2)

    # Parse arguments
    mode, kwargs = _parse_arguments(args_str)

    # Create processor
    if processor_type == 'mask':
        return MaskProcessor(mode=mode, **kwargs)
    elif processor_type == 'truncate':
        return TruncateProcessor(mode=mode, **kwargs)
    elif processor_type == 'shuffle':
        return ShuffleProcessor(mode=mode, **kwargs)
    elif processor_type == 'insert':
        return InsertProcessor(mode=mode, **kwargs)
    elif processor_type == 'question':
        return QuestionProcessor(mode=mode, **kwargs)
    elif processor_type == 'remove':
        return RemoveProcessor(mode=mode, **kwargs)
    elif processor_type == 'replace':
        # For replace, the first argument is 'pattern', not 'mode'
        return ReplaceProcessor(pattern=mode, **kwargs)
    elif processor_type == 'answer':
        return AnswerProcessor(mode=mode, **kwargs)
    elif processor_type == 'reason_is':
        # For reason_is, the first argument is 'text', not 'mode'
        return ReasonIsProcessor(text=mode, **kwargs)
    elif processor_type == 'random':
        return RandomProcessor(mode=mode, **kwargs)
    else:
        raise ValueError(f"Unknown processor type: {processor_type}")


def _parse_arguments(args_str: str) -> Tuple[str, Dict]:
    """
    Parse argument string into mode and kwargs

    Args:
        args_str: Argument string (e.g., "'number',mask_char='*',n=5")

    Returns:
        Tuple of (mode, kwargs)
    """
    if not args_str.strip():
        raise ValueError("Missing mode argument")

    # Split arguments by comma (respecting quotes)
    args = _split_arguments(args_str)

    if not args:
        raise ValueError("Missing mode argument")

    # First argument is the mode (must be a string)
    mode_arg = args[0].strip()
    mode = _parse_string_literal(mode_arg)

    # Remaining arguments are kwargs
    kwargs = {}
    for arg in args[1:]:
        arg = arg.strip()
        if '=' not in arg:
            raise ValueError(f"Invalid keyword argument: {arg}")

        key, value = arg.split('=', 1)
        key = key.strip()
        value = value.strip()

        # Parse value
        kwargs[key] = _parse_value(value)

    return mode, kwargs


def _split_arguments(args_str: str) -> List[str]:
    """
    Split argument string by commas, respecting quotes

    Args:
        args_str: Argument string

    Returns:
        List of argument strings
    """
    args = []
    current_arg = []
    in_quote = False
    quote_char = None

    for char in args_str:
        if char in ('"', "'") and not in_quote:
            in_quote = True
            quote_char = char
            current_arg.append(char)
        elif char == quote_char and in_quote:
            in_quote = False
            quote_char = None
            current_arg.append(char)
        elif char == ',' and not in_quote:
            # Argument separator
            args.append(''.join(current_arg))
            current_arg = []
        else:
            current_arg.append(char)

    # Add last argument
    if current_arg:
        args.append(''.join(current_arg))

    return args


def _parse_string_literal(value: str) -> str:
    """
    Parse a string literal (remove quotes)

    Args:
        value: String with quotes (e.g., "'number'" or '"number"')

    Returns:
        String without quotes

    Raises:
        ValueError: If not a valid string literal
    """
    value = value.strip()
    if (value.startswith("'") and value.endswith("'")) or \
       (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    else:
        raise ValueError(f"Expected string literal, got: {value}")


def _parse_value(value: str):
    """
    Parse a value (string, int, float, etc.)

    Args:
        value: Value string

    Returns:
        Parsed value (str, int, float)
    """
    value = value.strip()

    # Try string literal
    if (value.startswith("'") and value.endswith("'")) or \
       (value.startswith('"') and value.endswith('"')):
        return value[1:-1]

    # Try int
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Return as string
    return value
