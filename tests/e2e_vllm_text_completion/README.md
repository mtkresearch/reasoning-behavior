# vLLM Text Completion 端到端測試

## 概述

這個測試套件用於驗證 `LLMClient` 的 text completion 功能，支援在 Mac 上使用 mock server 測試，並可無縫轉移到 GPU 機器使用真實 vLLM server。

**測試涵蓋範圍：**
- ✅ Text completion API 基本功能
- ✅ 不同 model types 的 template 格式化
- ✅ 批次處理和並行執行
- ✅ Local mode 與 vLLM server 的整合

## 快速開始

### 場景 1：Mac 本地測試

```bash
# 方法 A：自動化測試（推薦）
source .venv/bin/activate
bash tests/e2e_vllm_text_completion/run_test.sh

# 方法 B：手動測試（適合除錯）
# Terminal 1: 啟動 mock server
python tests/e2e_vllm_text_completion/mock_vllm_server.py

# Terminal 2: 執行測試
python tests/e2e_vllm_text_completion/test_text_completion.py
```

### 場景 2：GPU 機器測試

```bash
# 步驟 1: 啟動 vLLM server（根據你的模型選擇）
python -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --port 8001 \
    --tensor-parallel-size 4

# 步驟 2: 測試 server
python tests/e2e_vllm_text_completion/test_real_vllm.py

# 步驟 3: 看到成功訊息
# ✓ All tests passed!
```

### 場景 3：在實驗中使用

```python
from llm_client import LLMClient, CompletionRequest

# 連接到 vLLM server
client = LLMClient(mode="local", base_url="http://localhost:8001/v1")

# 建立請求
request = CompletionRequest(
    question="What is 2 + 2?",
    reasoning="Let me calculate this.",
    answer_prefix="The answer is ",
    model_type="gpt-oss",
    temperature=0.7,
    max_tokens=200,
)

# 執行
result = client.complete(request)
print(result)
```

## 測試腳本說明

| 腳本 | 用途 | 使用場景 |
|------|------|---------|
| `mock_vllm_server.py` | Mock vLLM server | Mac 本地測試 |
| `test_text_completion.py` | 端到端測試（使用 mock） | Mac 本地測試 |
| `test_real_vllm.py` | 真實 vLLM server 測試 | GPU 機器測試 |
| `example_usage.py` | 使用範例程式 | 學習如何使用 |
| `run_test.sh` | 自動化測試腳本 | Mac 快速測試 |
| `quick_verify.py` | 快速驗證 mock server | 除錯用 |

## 支援的 Model Types

| Model Type | 實際模型 | Template Format |
|------------|---------|----------------|
| `gpt-oss` | openai/gpt-oss-120b | GPT-OSS format |
| `deepseek` | deepseek-ai/DeepSeek-V3 | DeepSeek Chat format |
| `deepseek-base` | deepseek-ai/DeepSeek-V3-Base | DeepSeek Base format |
| `olmo` | allenai/OLMo-3.1-32B-Think | ChatML format |

## 常用命令速查

### Mac 測試

```bash
# 啟動 mock server
python tests/e2e_vllm_text_completion/mock_vllm_server.py

# 自動化測試
bash tests/e2e_vllm_text_completion/run_test.sh

# 查看使用範例
python tests/e2e_vllm_text_completion/example_usage.py
```

### GPU 機器測試

```bash
# 啟動 vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model <model-name> \
    --port 8001 \
    --tensor-parallel-size <num-gpus>

# 測試本地 server
python tests/e2e_vllm_text_completion/test_real_vllm.py

# 測試遠端 server
python tests/e2e_vllm_text_completion/test_real_vllm.py \
    --base-url http://gpu-server:8001/v1 \
    --model-type gpt-oss

# 使用自訂問題測試
python tests/e2e_vllm_text_completion/test_real_vllm.py \
    --question "Calculate 123 * 456" \
    --reasoning "I will multiply step by step"
```

### 除錯

```bash
# 檢查 server 狀態
curl http://localhost:8001/v1/models

# 檢查 port 是否被佔用
lsof -i :8001

# 快速驗證 mock server
python tests/e2e_vllm_text_completion/quick_verify.py
```

## 問題排查

| 問題 | 可能原因 | 解決方法 |
|------|---------|---------|
| Connection refused | Server 未啟動 | 啟動 vLLM server 或 mock server |
| Port already in use | Port 被佔用 | `lsof -i :8001` 查看並關閉佔用的程序 |
| Model type mismatch | 參數不匹配 | 檢查 `--model-type` 是否與部署的模型匹配 |
| CUDA OOM | GPU 記憶體不足 | 增加 `--tensor-parallel-size` |
| Timeout | 請求超時 | 增加 `timeout` 參數或減少 `max_tokens` |
| API returned no choices | Template 格式錯誤 | 啟用 DEBUG mode 檢查 formatted prompt |

## 測試輸出範例

### 成功的測試（Mac）

```
============================================================
Test Summary
============================================================
✓ PASS: Basic Completion
✓ PASS: Different Question
✓ PASS: Local Mode Settings

============================================================
✓ All tests passed!
============================================================
```

### 成功的測試（GPU）

```
============================================================
vLLM Server Comprehensive Test
============================================================
✓ Server is healthy
✓ Available model: openai/gpt-oss-120b
✓ Completion successful! (took 2.34s)

============================================================
✓ All tests passed!
============================================================
```

---

## 附錄

### A. GPU 部署詳細步驟

#### 不同模型的啟動命令

**GPT-OSS (120B)**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --port 8001 \
    --tensor-parallel-size 4
```

**DeepSeek V3**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-V3 \
    --port 8001 \
    --tensor-parallel-size 8
```

**DeepSeek V3 Base**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-V3-Base \
    --port 8001 \
    --tensor-parallel-size 8
```

**OLMo 3.1**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model allenai/OLMo-3.1-32B-Think \
    --port 8001 \
    --tensor-parallel-size 2
```

#### 不同環境的 base_url 設定

```python
# Mac 本地測試（mock server）
client = LLMClient(mode="local", base_url="http://localhost:8001/v1")

# GPU 機器本地
client = LLMClient(mode="local", base_url="http://localhost:8001/v1")

# 遠端 GPU 機器（使用 IP）
client = LLMClient(mode="local", base_url="http://192.168.1.100:8001/v1")

# 遠端 GPU 機器（使用域名）
client = LLMClient(mode="local", base_url="http://gpu-server.example.com:8001/v1")
```

### B. API 端點說明

Mock server 和真實 vLLM server 都實現了以下標準 OpenAI-compatible API：

**GET `/v1/models`**
- 返回可用模型列表
- 用於驗證 server 健康狀態

**POST `/v1/completions`**
- 處理 text completion 請求
- 標準請求參數：
  - `prompt`: 完整的 prompt 文本
  - `model`: 模型名稱
  - `temperature`: 溫度參數 (0.0-2.0)
  - `max_tokens`: 最大生成 token 數
- 標準回應格式：
  ```json
  {
    "choices": [
      {
        "text": "completion text",
        "finish_reason": "stop"
      }
    ]
  }
  ```

**注意：** `reasoning` 欄位不是標準 vLLM API 的一部分，只有某些 OpenRouter 提供者才支援。

### C. 效能調優建議

#### 1. 批次處理

使用 `complete_concurrent()` 進行批次處理：

```python
from llm_client import Task, CompletionRequest

tasks = [
    Task(index=i, request=CompletionRequest(...))
    for i in range(100)
]

for completed_task in client.complete_concurrent(tasks, max_workers=10):
    print(f"Task {completed_task.index}: {completed_task.response.content}")
```

#### 2. Temperature 設定

- `temperature=0.0` - 確定性輸出，適合需要一致性的任務
- `temperature=0.7` - 平衡創造性與一致性（推薦）
- `temperature=1.0` - 更多創造性，適合開放性任務

#### 3. Max Tokens 設定

- 根據任務需求設定合適的 `max_tokens`
- 過大會浪費計算資源
- 過小可能截斷重要內容

#### 4. 啟用 Debug 模式

在 `llm_client.py` 中設定：
```python
DEBUG = True
```

或在程式中動態設定：
```python
import llm_client
llm_client.DEBUG = True
```

### D. 進階測試範例

#### 測試批次處理

```python
from llm_client import LLMClient, CompletionRequest, Task

client = LLMClient(mode="local", base_url="http://localhost:8001/v1")

questions = [
    "What is 1 + 1?",
    "What is 2 + 2?",
    "What is 3 + 3?",
]

tasks = []
for i, q in enumerate(questions):
    request = CompletionRequest(
        question=q,
        reasoning="Let me calculate.",
        answer_prefix="",
        model_type="gpt-oss",
        temperature=0.7,
        max_tokens=50,
    )
    tasks.append(Task(index=i, request=request))

for task in client.complete_concurrent(tasks, max_workers=3):
    if task.response.success:
        print(f"Task {task.index}: {task.response.content}")
```

#### 測試不同溫度設定

```python
temperatures = [0.0, 0.5, 1.0]

for temp in temperatures:
    request = CompletionRequest(
        question="Write a creative story opening.",
        reasoning="",
        answer_prefix="",
        model_type="gpt-oss",
        temperature=temp,
        max_tokens=100,
    )

    result = client.complete(request)
    print(f"\nTemperature {temp}: {result}")
```

#### 效能基準測試

```python
import time
from llm_client import LLMClient, CompletionRequest

client = LLMClient(mode="local", base_url="http://localhost:8001/v1")

num_tests = 5
total_time = 0

for i in range(num_tests):
    request = CompletionRequest(
        question="What is 2 + 2?",
        reasoning="Let me calculate.",
        answer_prefix="",
        model_type="gpt-oss",
        temperature=0.7,
        max_tokens=50,
    )

    start = time.time()
    result = client.complete(request)
    elapsed = time.time() - start
    total_time += elapsed

    print(f"Test {i+1}: {elapsed:.2f}s")

avg_time = total_time / num_tests
print(f"\nAverage time: {avg_time:.2f}s")
print(f"Throughput: {1/avg_time:.2f} requests/second")
```

### E. 依賴套件

```bash
# 安裝測試所需的套件
source .venv/bin/activate
uv pip install flask requests
```

### F. 參考資源

- [vLLM 官方文檔](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)
- [OpenAI API 規格](https://platform.openai.com/docs/api-reference/completions)
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
