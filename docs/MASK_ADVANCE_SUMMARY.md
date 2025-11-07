# Mask Numbers All-Advance Mode - 功能总结

## 概述

`all-advance` 模式是一种智能数字遮罩策略，用于在保留代数符号的同时遮罩计算性数字。

## 遮罩规则（优先级顺序）

### 1. 🔴 答案硬规则（最高优先级）

**如果数字等于答案，强制遮罩**

- 即使符合其他不遮罩规则，也必须遮罩
- 确保答案不会泄漏给模型

**示例：**
```
answer = "42"
x42       → x██        (紧邻字母仍遮)
n < 42    → n < ██     (不等式中仍遮)
A42       → A██        (变量索引仍遮)
```

### 2. ✅ 代数符号保护规则

**数字紧邻字母或下划线 → 不遮**

- **变量索引**：`A12`, `x1`, `var123` → 保留
- **下标**：`x_1`, `a_2`, `var_10` → 保留
- **系数**：`3x`, `5a`, `2y` → 保留
- **序数词**：`1st`, `2nd`, `3rd` → 保留

### 3. ✅ 不等式保护规则

**数字附近（10字符窗口内）有不等式符号 → 不遮**

支持的不等式符号：
- 基本符号：`<`, `>`, `≤`, `≥`, `≦`, `≧`
- ASCII 形式：`<=`, `>=`
- LaTeX 命令：`\leq`, `\geq`, `\le`, `\ge`, `\lt`, `\gt`

**示例：**
```
n < 5              → n < 5
1 ≤ x ≤ 10         → 1 ≤ x ≤ 10
a >= 100           → a >= 100
```

### 4. ❌ 乘法例外规则

**数字 x 数字 → 强制遮罩**

处理 `3x3` 这类乘法表达式（即使 `x` 是字母）

**示例：**
```
3x3       → █x█
2x5       → █x█
10x10     → ██x██
```

### 5. ❌ 其他数字全部遮罩

**默认遮罩所有不符合上述规则的数字**

- **次方**：`x^2` → `x^█`
- **函数参数**：`f(3)` → `f(█)`
- **运算**：`1+2` → `█+█`
- **等式**：`x = 20` → `x = ██`

## 函数签名

```python
def mask_numbers_all_advance(
    reasoning: str, 
    answer: str = None, 
    mask_char: str = '█'
) -> str
```

**参数：**
- `reasoning`: 原始推理文本
- `answer`: 正确答案（可选，如果提供则会强制遮罩）
- `mask_char`: 遮罩字符（默认：'█'）

**返回：**
- 遮罩后的推理文本

## 使用示例

### 基本用法（无答案）

```python
from mask_numbers_experiment import mask_numbers_all_advance

text = "Let x1 = 10 and x2 = 20, then x1 + x2 = 30"
result = mask_numbers_all_advance(text)
# 结果：Let x1 = ██ and x2 = ██, then x1 + x2 = ██
```

### 带答案（硬规则）

```python
text = "Let x1 = 10 and x2 = 20, then x1 + x2 = 30"
result = mask_numbers_all_advance(text, answer="30")
# 结果：Let x1 = ██ and x2 = ██, then x1 + x2 = ██
#       注意：即使 30 是结果，也会被遮罩
```

### 实际应用（在实验脚本中）

```bash
python mask_numbers_experiment.py \
  --results_path data/AIME2025__R10/gpt-oss/p1/results.json \
  --output_path data/baseline/mask_all_advance.json \
  --mask-mode all-advance \
  --model_type gpt-oss
```

## 测试覆盖

总计 **64 个测试用例**，全部通过 ✅

### 测试文件

1. **`tests/test_mask_advance.py`** - 46个测试
   - 基础规则测试：34个
   - 答案硬规则测试：12个

2. **`tests/test_mask_advance_example.py`** - 真实数学推理示例
   - 展示完整的遮罩效果
   - 验证答案 2400 正确遮罩

3. **`tests/test_mask_inequality.py`** - 18个不等式专项测试
   - 各种不等式格式
   - 边界情况测试

### 运行测试

```bash
# 从项目根目录
source .venv/bin/activate

# 运行所有测试
python tests/test_mask_advance.py
python tests/test_mask_advance_example.py
python tests/test_mask_inequality.py
```

## 规则优先级总结

```
1. answer == number → 遮 (最高优先级)
2. [A-Za-z_] 紧邻 number → 不遮
3. inequality 附近有 number → 不遮
4. digit x digit → 遮 (特殊规则)
5. 其他 → 遮 (默认)
```

## 版本历史

- **v1.0** - 基础规则：代数符号保护
- **v1.1** - 增加不等式保护规则
- **v1.2** - 增加答案硬规则（最高优先级）

---

**最后更新**: 2025-11-07
**测试状态**: ✅ 64/64 通过
