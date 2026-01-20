source .venv/bin/activate

# 定義共用的結果路徑
RESULTS_PATH="data/CodeElo/olmo/p1/results.json"


# 定義 flow 列表
FLOWS=(
  "shuffle('line'),answer('retrieval')"
  "answer('retrieval')"
)

# 執行每個 flow
for flow in "${FLOWS[@]}"; do
  echo "========================================="
  echo "Running flow: $flow"
  echo "========================================="
  uv run run_experiment.py --results_path "$RESULTS_PATH" --flow "$flow" --max_workers 4 
  echo ""
done

echo "All flows completed!"