source .venv/bin/activate

# 定義共用的結果路徑
RESULTS_PATH="data/CodeElo/gpt-oss/p1/results.json"


# 定義 flow 列表
FLOWS=(
  "truncate('all'),answer('retrieval')"
  "padding('token',tokenizer_model='deepseek-ai/DeepSeek-V3.1'),answer('retrieval')"
  "padding('word',words_tsv_path='data/AIME2025__R10/deepseek/p1/words.tsv'),answer('retrieval')"
)

# 執行每個 flow
for flow in "${FLOWS[@]}"; do
  echo "========================================="
  echo "Running flow: $flow"
  echo "========================================="
  uv run run_experiment.py --results_path "$RESULTS_PATH" --flow "$flow"
  echo ""
done

echo "All flows completed!"