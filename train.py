"""
Training script for code generation.

Usage:
```bash
python train.py \
    --model_name 'microsoft/Phi-3-mini-4k-instruct' \
    --epochs 1 \
    --learning_rate 2e-5 \
    --batch_size 1 \
    --desired_batch_size 8 \
    --lora_rank 32 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --exp_id justtest
```
"""

############################################################################################
#################################### Import libraries ######################################
############################################################################################
import os
import math
import time
import wandb
import torch
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import DataCollatorForSeq2Seq

from transformers import get_scheduler

from peft import LoraConfig, TaskType
from peft import get_peft_model

from utils.args import args
from utils.evaluation import evaluate
from utils.datasets import get_code_alpaca_20k
from utils.completions import clean_completion, inference


############################################################################################
######################################### LOGGING ##########################################
############################################################################################
os.environ["WANDB_SILENT"] = "true"
wandb_project = 'cody'
run = wandb.init(project=wandb_project, name=args.exp_id, config=args)
print(f'Run name: {run.name}. Visit at {run.get_url()}')


############################################################################################
############################ Set Constants: Model, Tokenizer & Data ########################
############################################################################################

# Model
model_name = args.model_name

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)
print(f"Model loaded: {model_name}")

## PEFT
lora_config = LoraConfig(
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=[
        "q_proj", "o_proj", "k_proj", "v_proj",
        "gate_proj", "up_proj", "down_proj", "dense"
    ],
    task_type=TaskType.CAUSAL_LM,
)

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=lora_model)
max_length = 1024
print(f"Tokenizer loaded: {model_name}")


## Tokenizer function
def tokenize_function(examples):
    # Tokenize the codes
    tokenized_outputs = tokenizer(
        examples["code"],
        truncation=True,
        max_length=max_length
    ) # , padding='max_length'
    
    # Set labels to input_ids. This assumes a task like text generation where
    # the model learns to predict the input sequence itself (next word).
    # You don’t need labels (also known as an unsupervised task)
    # because the next word is the label
    tokenized_outputs["labels"] = tokenized_outputs["input_ids"].copy()
    return tokenized_outputs


# Dataset: Code Alpaca
batch_size = args.batch_size  # number of examples in each batch

base_dataset = get_code_alpaca_20k()
tokenized_dataset = base_dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.with_format("torch")

split_dataset = tokenized_dataset["train"].train_test_split(
    test_size=args.test_size, shuffle=True, seed=42
)

## Pytorch dataloader format
# Remove columns that can't be converted to tensors
dataset = split_dataset.remove_columns([
    'output', 'input', 'instruction', 'code', 'prompt', 'completion'
])

# Move the data to tensors
dataset.set_format("torch")

train_dataloader = DataLoader(
    dataset["train"],  # For testing purposes ->  .shuffle(seed=42).select(range(1000)),
    shuffle=True, batch_size=batch_size,
    collate_fn=data_collator
)

print(f"Training dataset size: {len(dataset['train'])}")

test_dataloader = DataLoader(
    dataset["test"],  # For testing purposes -> .shuffle(seed=42).select(range(1000)),
    batch_size=batch_size,
    collate_fn=data_collator
)

print(f"Testing dataset size: {len(dataset['test'])}")


############################################################################################
#################################### Training setup ########################################
############################################################################################

learning_rate = args.learning_rate
weight_decay = 0.01

num_epochs = args.epochs

# The desired batch size is the batch size you want to train with
desired_batch_size = args.desired_batch_size
gradient_accumulation_steps = desired_batch_size // batch_size

# We set the maximum number of iterations to those ones needed to go through the dataset
# num_epochs times, considering the gradient accumulation
max_iters = num_epochs * len(train_dataloader) // gradient_accumulation_steps
eval_interval = args.eval_interval

print(f"Max iterations: {max_iters}")
print(f"Evaluation interval: {eval_interval}")
      
warmup_steps_ratio = 0.1
warmup_steps = math.ceil(max_iters * warmup_steps_ratio)


## Optimizer and learning rate scheduler
# Create an optimizer and learning rate scheduler to fine-tune the model.
# Let’s use the AdamW optimizer from PyTorch
optimizer = AdamW(lora_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Create the default learning rate scheduler from Trainer
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="cosine", optimizer=optimizer,
    num_warmup_steps=warmup_steps, num_training_steps=max_iters
)

## Training loop
# Iterating over the dataset in batches
progress_bar = tqdm(range(max_iters))
train_start = time.time()
for iter_num in range(max_iters):
    model.train()
    
    for micro_step in range(gradient_accumulation_steps):
        # Extract a batch of data
        batch = next(iter(train_dataloader))

        outputs = lora_model(**batch)
        # El modelo calcula su loss, pero podriamos acceder a los logits del modelo
        # y las labels del batch y calcular nuestra loss propia
        # scale the loss to account for gradient accumulation
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    progress_bar.update(1)

    if iter_num % eval_interval == 0:

        # scale up to undo the division above
        # approximating total loss (exact would have been a sum)
        train_loss = loss.item() * gradient_accumulation_steps

        test_loss, test_perplexity = evaluate(model, test_dataloader)

        print(f"### ITER {iter_num} ###")
        #print(f"Train Loss: {train_loss:.4f} - Train Perplexity: {train_perplexity:.4f}")
        print(f"Test Loss: {test_loss:.4f} - Test Perplexity: {test_perplexity:.4f}")

        wandb.log({
            "iter": iter_num,
            "train/loss": train_loss,
            "val/loss": test_loss,
            "val/perplexity": test_perplexity,
            "lr": lr_scheduler.get_last_lr()[0],
        })

progress_bar.close()
print(f"Training took: {time.time() - train_start:.2f} seconds")


############################################################################################
################################### Testing the Model ######################################
############################################################################################

test_samples = 25
assert test_samples <= len(dataset['test']), "Not enough samples for testing"

test_table = wandb.Table(columns=["Prompt", "Completion", "Model Completion"])

for case in range(test_samples):
    full_text = split_dataset['test'][case]['code']
    prompt_text = split_dataset['test'][case]['prompt']
    completion_text = split_dataset['test'][case]['completion']

    response = inference(prompt_text, lora_model, tokenizer, max_output_tokens=256)
    clean_response = clean_completion(response, tokenizer.eos_token, prompt_text)
    
    test_table.add_data(prompt_text, completion_text, clean_response)
    
# log the table to wandb
run.log({"test_completions": test_table})


############################################################################################
#################################### Save the Model ########################################
############################################################################################

lora_model.save_pretrained(f"checkpoints/{run.name}")

artifact = wandb.Artifact(f"model_{args.exp_id}", type="model_checkpoint")
artifact.add_dir(f"checkpoints/{run.name}")
run.log_artifact(artifact)

wandb.finish()
