set -e

source .venv/bin/activate

export TOKENIZERS_PARALLELISM=true

# 定義所有 p1/results.json 路徑列表
RESULTS_PATHS=(
  "data/AIME2025__R10/gpt-oss/p1/results.json"
  # "data/AIME2025__R10/olmo/p1/results.json"
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
  "insert('fix',sentence='Thus answer: 123.',count='100% # of answer'),shuffle('word'),answer('retrieval')"
  "insert('fix',sentence='Thus answer: 123.',count='200% # of answer'),shuffle('word'),answer('retrieval')"
  "insert('fix',sentence='Thus answer: 123.',count='300% # of answer'),shuffle('word'),answer('retrieval')"

  "insert('fix',sentence='Thus answer: 123.',count='100% # of answer'),mask('alphabet',mask_char=' '),replace('\s+',replacement=' '),shuffle('word'),answer('retrieval')"
  "insert('fix',sentence='Thus answer: 123.',count='200% # of answer'),mask('alphabet',mask_char=' '),replace('\s+',replacement=' '),shuffle('word'),answer('retrieval')"
  "insert('fix',sentence='Thus answer: 123.',count='300% # of answer'),mask('alphabet',mask_char=' '),replace('\s+',replacement=' '),shuffle('word'),answer('retrieval')"

  # "insert('fix',sentence='Thus answer: 123.',count='100% # of answer'),answer('retrieval')"
  # "insert('fix',sentence='Thus answer: 123.',count='200% # of answer'),answer('retrieval')"
  # "insert('fix',sentence='Thus answer: 123.',count='300% # of answer'),answer('retrieval')"
  # "shuffle('line'),insert('fix',sentence='Thus answer: 123.',count='100% # of answer'),answer('retrieval')"
  # "shuffle('line'),insert('fix',sentence='Thus answer: 123.',count='200% # of answer'),answer('retrieval')"
  # "shuffle('line'),insert('fix',sentence='Thus answer: 123.',count='300% # of answer'),answer('retrieval')"

  # "insert('fix',sentence='Thus answer: 123.',count='100% # of answer'),mask('alphabet',mask_char=' '),replace('\s+',replacement=' '),answer('retrieval')"
  # "insert('fix',sentence='Thus answer: 123.',count='200% # of answer'),mask('alphabet',mask_char=' '),replace('\s+',replacement=' '),answer('retrieval')"
  # "insert('fix',sentence='Thus answer: 123.',count='300% # of answer'),mask('alphabet',mask_char=' '),replace('\s+',replacement=' '),answer('retrieval')"
  # "shuffle('line'),insert('fix',sentence='Thus answer: 123.',count='100% # of answer'),mask('alphabet',mask_char=' '),replace('\s+',replacement=' '),answer('retrieval')"
  # "shuffle('line'),insert('fix',sentence='Thus answer: 123.',count='200% # of answer'),mask('alphabet',mask_char=' '),replace('\s+',replacement=' '),answer('retrieval')"
  # "shuffle('line'),insert('fix',sentence='Thus answer: 123.',count='300% # of answer'),mask('alphabet',mask_char=' '),replace('\s+',replacement=' '),answer('retrieval')"


  # "mask('alphabet',mask_char=' '),replace('\s+',replacement=' '),answer('retrieval')"
  # "shuffle('line'),truncate('answer'),mask('alphabet',mask_char=' '),replace('\s+',replacement=' '),answer('retrieval')"

  # "shuffle('line'),mask('alphabet',mask_char=' '),replace('\s+',replacement=' '),answer('retrieval')"
  # "shuffle('word'),mask('alphabet',mask_char=' '),replace('\s+',replacement=' '),answer('retrieval')"

  # "question('remove'),mask('alphabet'),answer('retrieval')"

  # "shuffle('line'),answer('retrieval')"
  # "mask('alphabet'),shuffle('line'),answer('retrieval')"
  # "question('remove'),shuffle('line'),answer('retrieval')"

  # ""
  # "mask('alphabet'),answer('retrieval')"
  # "mask('answer'),answer('retrieval')"
  # "mask('alphabet'),mask('answer'),answer('retrieval')"
  # "mask('number'),answer('retrieval')"
  # "question('remove'),answer('retrieval')"
  # "question('remove'),mask('alphabet'),answer('retrieval')"
  # "question('remove'),mask('alphabet'),mask('answer'),answer('retrieval')"

  # "mask('alphabet',mask_char=' '),replace('\s+',replacement=' '),answer('retrieval')"
  # "mask('alphabet',mask_char=' '),replace('\s+',replacement=' '),truncate('answer'),answer('retrieval')"
  # "question('remove'),mask('number'),answer('retrieval')"
  # "question('remove'),mask('answer'),shuffle('line'),answer('retrieval')"
  # "question('remove'),mask('alphabet'),mask('answer'),shuffle('line'),answer('retrieval')"
  # "question('remove'),mask('number'),shuffle('line'),answer('retrieval')"

  # "question('remove'),mask('answer'),answer('retrieval')"
  # "mask('number'),shuffle('line'),answer('retrieval')"
  # "mask('alphabet'),shuffle('line'),answer('retrieval')"
  # "mask('answer'),shuffle('line'),answer('retrieval')"
  # "mask('alphabet'),mask('answer'),shuffle('line'),answer('retrieval')"
  # "question('remove'),mask('alphabet'),shuffle('line'),answer('retrieval')"

  # "shuffle('token',model_type='deepseek'),answer('retrieval')"
  # "padding('token',tokenizer_model='deepseek-ai/DeepSeek-V3'),answer('retrieval')"
  # "padding('word',words_tsv_path='data/CodeElo/deepseek/p1/words.tsv'),answer('retrieval')"

  # ""
  # "shuffle('line'),answer('retrieval')"
  # "shuffle('line'),truncate('answer'),answer('retrieval')"
  # "truncate('all'),answer('retrieval')"
  # "answer('retrieval')"
  # "shuffle('word'),answer('retrieval')"
  # "shuffle('in-line-word'),answer('retrieval')"
  # "shuffle('in-line-word'),truncate('answer'),answer('retrieval')"
  # "shuffle('in-line-word'),shuffle('line'),answer('retrieval')"
  # "truncate('all')"
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
    uv run run_experiment.py --results_path "$results_path" --flow "$flow" --max_workers 16

    echo ""
  done

  echo ""
done

echo "All flows completed for all files!"
