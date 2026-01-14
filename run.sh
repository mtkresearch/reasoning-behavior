source .venv/bin/activate

# 定義共用的結果路徑
RESULTS_PATH="data/AIME2025__R10/deepseek/p1/results.json"

# 定義 flow 列表
FLOWS=(
  "answer('retrieval')"
  "truncate('all'),answer('retrieval')"
  "shuffle('word'),answer('retrieval')"
  "shuffle('line'),answer('retrieval')"
  "shuffle('in-line-word'),answer('retrieval')"
  "shuffle('in-line-word'),shuffle('line'),answer('retrieval')"
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