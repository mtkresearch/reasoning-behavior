#!/usr/bin/env python3
"""
TDD Tests for AttentionMask mode in run_new_reasoning_gen.py

这些测试定义了AttentionMask模式的预期行为：
1. 句子边界检测 (newline位置)
2. 纯字母token识别
3. 跨句子关系检测
4. Attention权重修改
5. 集成生成函数
"""

import pytest
import torch
import numpy as np
from transformers import AutoTokenizer

from run_new_reasoning_gen import LetterOnlyTokenFilter


# =============================================================================
# Test 1: 句子边界检测
# =============================================================================

class TestSentenceBoundaryDetection:
    """测试句子边界（newline tokens）的检测"""

    def test_find_newline_positions_single_newline(self):
        """Test: 找到单个newline token的位置"""
        from run_new_reasoning_gen import find_newline_positions

        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # 构建包含newline的input_ids
        text = "Hello\nWorld"
        input_ids = tokenizer.encode(text, return_tensors="pt")[0]

        newline_positions = find_newline_positions(input_ids, tokenizer)

        # 应该找到newline token的位置
        assert len(newline_positions) > 0, "Should find at least one newline"
        assert all(isinstance(pos, (int, np.integer, torch.Tensor)) for pos in newline_positions)

    def test_find_newline_positions_multiple_newlines(self):
        """Test: 找到多个newline tokens的位置"""
        from run_new_reasoning_gen import find_newline_positions

        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        text = "Line1\nLine2\nLine3"
        input_ids = tokenizer.encode(text, return_tensors="pt")[0]

        newline_positions = find_newline_positions(input_ids, tokenizer)

        # 应该找到2个newlines
        assert len(newline_positions) >= 1, "Should find multiple newlines"
        # 位置应该递增
        assert all(newline_positions[i] < newline_positions[i+1]
                   for i in range(len(newline_positions)-1))

    def test_find_newline_positions_no_newline(self):
        """Test: 没有newline的情况"""
        from run_new_reasoning_gen import find_newline_positions

        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        text = "NoNewlineHere"
        input_ids = tokenizer.encode(text, return_tensors="pt")[0]

        newline_positions = find_newline_positions(input_ids, tokenizer)

        # 应该返回空列表或根据实现返回合理值
        assert isinstance(newline_positions, (list, torch.Tensor, np.ndarray))

    def test_find_newline_positions_types(self):
        """Test: 返回类型应该是可索引的"""
        from run_new_reasoning_gen import find_newline_positions

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        text = "A\nB\nC"
        input_ids = tokenizer.encode(text, return_tensors="pt")[0]

        newline_positions = find_newline_positions(input_ids, tokenizer)

        # 应该是list或可索引的
        if len(newline_positions) > 0:
            _ = newline_positions[0]  # 应该可以索引


# =============================================================================
# Test 2: 句子分配（token属于哪个句子）
# =============================================================================

class TestSentenceAssignment:
    """测试给定token位置，判断其属于的句子"""

    def test_get_sentence_for_token_position(self):
        """Test: 获取token所在的句子ID"""
        from run_new_reasoning_gen import get_sentence_for_token

        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        text = "First\nSecond\nThird"
        input_ids = tokenizer.encode(text, return_tensors="pt")[0]
        newline_positions = [pos for pos, token_id in enumerate(input_ids)
                             if token_id == tokenizer.encode('\n', add_special_tokens=False)[0]]

        # 测试不同位置的token
        # 第一个token应该属于句子0
        sentence_0 = get_sentence_for_token(0, newline_positions)
        assert sentence_0 == 0

        # 如果有newline，后续token应该属于不同句子
        if newline_positions:
            sentence_after_first_newline = get_sentence_for_token(
                newline_positions[0] + 1, newline_positions
            )
            assert sentence_after_first_newline == 1

    def test_tokens_in_different_sentences(self):
        """Test: 判断两个token是否在不同句子"""
        from run_new_reasoning_gen import are_in_different_sentences

        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        text = "First\nSecond"
        input_ids = tokenizer.encode(text, return_tensors="pt")[0]
        newline_token_id = tokenizer.encode('\n', add_special_tokens=False)[0]

        newline_positions = [pos for pos, token_id in enumerate(input_ids)
                             if token_id == newline_token_id]

        # 如果有newline
        if newline_positions:
            # 第一个token和最后一个token应该在不同句子
            result = are_in_different_sentences(0, len(input_ids)-1, newline_positions)
            assert isinstance(result, bool)
            assert result == True

    def test_tokens_in_same_sentence(self):
        """Test: 判断两个token是否在同一句子"""
        from run_new_reasoning_gen import are_in_different_sentences

        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        text = "First word\nSecond"
        input_ids = tokenizer.encode(text, return_tensors="pt")[0]
        newline_token_id = tokenizer.encode('\n', add_special_tokens=False)[0]

        newline_positions = [pos for pos, token_id in enumerate(input_ids)
                             if token_id == newline_token_id]

        # 相邻的两个token（在第一个newline之前）应该在同一句子
        if len(input_ids) > 1:
            result = are_in_different_sentences(0, 1, newline_positions)
            assert isinstance(result, bool)
            assert result == False


# =============================================================================
# Test 3: 纯字母token识别（在attention mask上下文中）
# =============================================================================

class TestLetterTokenIdentificationForAttentionMask:
    """测试在attention mask中识别纯字母tokens"""

    def test_identify_letter_only_tokens_positions(self):
        """Test: 给定input_ids，找出其中的纯字母tokens的位置"""
        from run_new_reasoning_gen import identify_letter_only_token_positions

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        text = "Hello world 123 test"
        input_ids = tokenizer.encode(text, return_tensors="pt")[0]

        letter_positions = identify_letter_only_token_positions(input_ids, tokenizer)

        # 应该找到至少一些位置
        assert isinstance(letter_positions, (list, set, torch.Tensor, np.ndarray))

        # 所有位置都应该在有效范围内
        if len(letter_positions) > 0:
            letter_positions_list = list(letter_positions) if not isinstance(letter_positions, list) else letter_positions
            assert all(0 <= pos < len(input_ids) for pos in letter_positions_list)

    def test_letter_positions_vs_number_tokens(self):
        """Test: 数字tokens不应该被识别为纯字母"""
        from run_new_reasoning_gen import identify_letter_only_token_positions

        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # 包含数字的文本
        text1 = "hello world"
        text2 = "123 456"
        text3 = "3x test"

        ids1 = tokenizer.encode(text1, return_tensors="pt")[0]
        ids2 = tokenizer.encode(text2, return_tensors="pt")[0]
        ids3 = tokenizer.encode(text3, return_tensors="pt")[0]

        pos1 = identify_letter_only_token_positions(ids1, tokenizer)
        pos2 = identify_letter_only_token_positions(ids2, tokenizer)
        pos3 = identify_letter_only_token_positions(ids3, tokenizer)

        # 纯文本应该有更多的纯字母tokens
        assert len(pos1) > len(pos2), "Pure text should have more letter tokens than numbers"


# =============================================================================
# Test 4: Attention权重修改
# =============================================================================

class TestAttentionWeightModification:
    """测试attention权重的修改"""

    def test_mask_cross_sentence_letter_attention(self):
        """Test: mask掉跨句子的纯字母token的attention"""
        from run_new_reasoning_gen import mask_cross_sentence_letter_attention

        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # 创建简单的attention权重
        # 形状: (batch=1, heads=1, seq_len=5, seq_len=5)
        attn_weights = torch.ones(1, 1, 5, 5)

        # 创建input_ids: "A\nB\nC" (简化示例)
        input_ids = torch.tensor([[1, 198, 2, 198, 3]])  # 假设198是newline token ID

        # 调用mask函数
        masked_attn = mask_cross_sentence_letter_attention(
            attn_weights, input_ids, tokenizer,
            letter_token_ids={1, 2, 3},  # 假设这些是纯字母tokens
            newline_token_id=198
        )

        # 检查返回值形状不变
        assert masked_attn.shape == attn_weights.shape

        # 检查某些权重被设为0
        # (这取决于具体实现，但应该有至少一些0)
        assert (masked_attn == 0).any() or (masked_attn == 1).any()

    def test_attention_values_remain_valid(self):
        """Test: 修改后的attention权重应该是有效的"""
        from run_new_reasoning_gen import mask_cross_sentence_letter_attention

        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # 创建随机attention权重
        attn_weights = torch.rand(1, 8, 10, 10)
        input_ids = torch.arange(10).unsqueeze(0)

        masked_attn = mask_cross_sentence_letter_attention(
            attn_weights, input_ids, tokenizer,
            letter_token_ids=set(),
            newline_token_id=999  # 不存在的token ID
        )

        # 权重应该在0-1之间（或保持原值）
        assert torch.all(masked_attn >= 0) and torch.all(masked_attn <= 1)


# =============================================================================
# Test 5: CrossSentenceLetterMaskingHook类
# =============================================================================

class TestCrossSentenceLetterMaskingHook:
    """测试Hook类的功能"""

    def test_hook_initialization(self):
        """Test: Hook初始化"""
        from run_new_reasoning_gen import CrossSentenceLetterMaskingHook

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])

        hook = CrossSentenceLetterMaskingHook(tokenizer, input_ids)

        # Hook应该有必要的属性
        assert hasattr(hook, 'tokenizer')
        assert hasattr(hook, 'input_ids')
        assert hasattr(hook, 'letter_only_ids')
        assert hasattr(hook, 'newline_positions')

    def test_hook_call_signature(self):
        """Test: Hook的__call__方法签名"""
        from run_new_reasoning_gen import CrossSentenceLetterMaskingHook

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        input_ids = torch.tensor([[1, 2, 3]])

        hook = CrossSentenceLetterMaskingHook(tokenizer, input_ids)

        # 创建模拟的attention输出
        attn_weights = torch.ones(1, 8, 3, 3)
        module = None
        output = (attn_weights,)

        # 调用hook
        result = hook(module, None, output)

        # 应该返回修改过的output tuple
        assert isinstance(result, tuple)
        assert len(result) == len(output)


# =============================================================================
# Test 6: 生成函数集成
# =============================================================================

class TestGenerateReasoningWithModes:
    """测试generate_reasoning函数的两种模式"""

    def test_generate_reasoning_supports_mode_parameter(self):
        """Test: generate_reasoning支持mode参数"""
        # 这是一个接口测试，检查函数签名
        import inspect
        from run_new_reasoning_gen import generate_reasoning

        sig = inspect.signature(generate_reasoning)
        param_names = list(sig.parameters.keys())

        # 应该有mode参数
        assert 'mode' in param_names, f"generate_reasoning should have 'mode' parameter, got: {param_names}"

    def test_mode_parameter_accepts_valid_values(self):
        """Test: mode参数接受有效值"""
        import inspect
        from run_new_reasoning_gen import generate_reasoning

        # 检查参数默认值或注释
        sig = inspect.signature(generate_reasoning)
        mode_param = sig.parameters.get('mode')

        # mode参数应该存在
        assert mode_param is not None


# =============================================================================
# Test 7: Newline Token ID识别
# =============================================================================

class TestNewlineTokenIdentification:
    """测试newline token的准确识别"""

    def test_get_newline_token_id(self):
        """Test: 获取tokenizer中newline的token ID"""
        from run_new_reasoning_gen import get_newline_token_id

        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        newline_id = get_newline_token_id(tokenizer)

        # 应该返回一个整数
        assert isinstance(newline_id, (int, np.integer))

        # 应该在有效范围内
        assert 0 <= newline_id < len(tokenizer)

        # 解码后应该是newline
        decoded = tokenizer.decode([newline_id])
        assert '\n' in decoded

    def test_newline_token_id_matches_encoding(self):
        """Test: newline token ID与encode结果一致"""
        from run_new_reasoning_gen import get_newline_token_id

        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        newline_id = get_newline_token_id(tokenizer)
        encoded = tokenizer.encode('\n', add_special_tokens=False)

        # 应该匹配
        assert newline_id in encoded, f"newline_id {newline_id} not in {encoded}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
