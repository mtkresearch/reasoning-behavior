set -e

source .venv/bin/activate

# 定義所有 p1/results.json 路徑列表
RESULTS_PATHS=(
  # "data/AIME2025__R10/gpt-oss/p1/results.json"
  "data/AIME2025__R10/olmo/p1/results.json"
  # "data/AIME2025__R10/deepseek/p1/results.json"
  # "data/GPQA-Diamond/gpt-oss/p1/results.json"
  # "data/GPQA-Diamond/olmo/p1/results.json"
  # "data/GPQA-Diamond/deepseek/p1/results.json"
  # "data/CodeElo/gpt-oss/p1/results.json"
  # "data/CodeElo/olmo/p1/results.json"
  # "data/CodeElo/deepseek/p1/results.json"
)

# 定義 flow 列表
FLOWS=(
  # ""
  "truncate('all'),answer('retrieval')"
  # "truncate('all')"
  # "shuffle('line'),answer('retrieval')"
  # "answer('retrieval')"
  # "shuffle('word'),answer('retrieval')"
  # "shuffle('in-line-word'),answer('retrieval')"
  # "shuffle('in-line-word'),shuffle('line'),answer('retrieval')"
)

# 對每個結果檔案執行所有 flow
for results_path in "${RESULTS_PATHS[@]}"; do
  echo "###############################################"
  echo "Processing: $results_path"
  echo "###############################################"
  echo ""

  # 執行每個 flow
  for flow in "${FLOWS[@]}"; do
    echo "========================================="
    echo "Running flow: $flow"
    echo "========================================="
    uv run run_experiment.py --results_path "$results_path" --flow "$flow" --max_workers 2
    echo ""
  done

  echo ""
done

echo "All flows completed for all files!"
