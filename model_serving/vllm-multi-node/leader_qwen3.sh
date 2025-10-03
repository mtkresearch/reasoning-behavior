MODEL=/mnt/shared/p01/yc/models/Qwen3-235B-A22B-Thinking-2507

./multi-node-serving.sh leader --ray_port=6779 --ray_cluster_size=2 && \
  vllm serve $MODEL --port 8001 --tensor-parallel-size 8 --pipeline_parallel_size 2 --max-model-len 262144
