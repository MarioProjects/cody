import uuid
import itertools
from lightning_sdk import Studio, Machine

# reference to the current studio
# if you run outside of Lightning, you can pass the Studio name
studio = Studio()

# use the jobs plugin
studio.install_plugin('jobs')
job_plugin = studio.installed_plugins['jobs']

# do a sweep over learning rates
models = [
    {"model": 'Qwen/Qwen1.5-0.5B-Chat', "batch_size": 4},
    {"model": 'google/gemma-2b-it', "batch_size": 2},
    {"model": 'microsoft/Phi-3-mini-4k-instruct', "batch_size": 2}
]

epochs = [5, 10, 15]
learning_rate = [2e-5, 2e-4]
desired_batch_size = [8, 16]

lora_rank = [32, 64]
lora_alpha = [16, 32]
lora_dropout = [0.1]

combinations = list(itertools.product(
    models,
    epochs, learning_rate, desired_batch_size,
    lora_rank, lora_alpha, lora_dropout
))

print(f"Total combinations: {len(combinations)}")

# start all jobs on an A10G GPU with names containing an index
for combination in combinations:
    experiment_id = str(uuid.uuid4())
    model_name = combination[0]['model']
    batch_size = combination[0]['batch_size']
    cmd = f"""python train.py \
        --model_name {model_name} \
        --epochs {combination[1]} \
        --learning_rate {combination[2]} \
        --batch_size {batch_size} \
        --desired_batch_size {combination[3]} \
        --lora_rank {combination[4]} \
        --lora_alpha {combination[5]} \
        --lora_dropout {combination[6]} \
        --exp_id {experiment_id}
    """

    job_name = f'run-{experiment_id}'
    job_plugin.run(cmd, machine=Machine.T4, name=job_name)
