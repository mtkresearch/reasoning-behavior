uv run run_experiment.py --flow "mask('all-nonblank')"; 

# uv run run_experiment.py --flow "question('remove'),truncate('answer_and_after'),mask('alphabet'),shuffle('line')";

# uv run run_experiment.py --flow "truncate('answer_and_after'),mask('alphabet'),question('remove')"; 
# uv run run_experiment.py --flow "truncate('answer_and_after'),mask('alphabet'),question('remove'),shuffle('line')"; 