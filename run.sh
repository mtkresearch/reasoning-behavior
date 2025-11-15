# uv run run_experiment.py --flow "mask('alphabet',mask_char=' '),replace('\\s+',replacement=' ')" ; 

uv run run_experiment.py --flow "mask('alphabet',mask_char=' '),replace(' +',replacement=' '),shuffle('line')"; 

# uv run run_experiment.py --flow "question('remove'),truncate('answer_and_after'),mask('alphabet'),shuffle('line')";

# uv run run_experiment.py --flow "truncate('answer_and_after'),mask('alphabet'),question('remove')"; 
# uv run run_experiment.py --flow "truncate('answer_and_after'),mask('alphabet'),question('remove'),shuffle('line')"; 