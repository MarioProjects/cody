# Cody

![Welcome Illustration](assets/welcome.jpg "Welcome Illustration")

The purpose of this project is to tackle the finetuning of a Large Language Model (LLM) on the task of code generation.

# Index

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Models](#models)
4. [Datasets](#datasets)
5. [Evaluation: HumanEval](#evaluation-humaneval)

# Project Overview

The repository is structured as follows:

```
├── train.py       <- Training the model
├── run_search.py  <- Searching hyperparameters Lightning AI
├── utils          <- Utility functions
│   ├── args.py         <- Arguments parser
│   ├── completions.py  <- Completions generation
│   ├── data.py         <- Data pre/post-processing
│   ├── datasets.py     <- Dataset loading
│   └── evaluation.py   <- Evaluation functions
|
└── notebooks  <- Jupyter notebooks
    ├── data_preparation.ipynb  <- Data preparation
    └── training.ipynb          <- Training the model
```

# Installation

TBD

# Models

We will use the `transformers` library to finetune a model on the task of code generation. As initial step I benchmark the initial performance, without any finetuning, of several models on the HumanEval dataset:

| Model Name             | Number Parameters | Generations | HumanEval |
|------------------------|-------------------|-------------|-----------|
| microsoft/phi-1        | 1.3B              | 1           | 0.48      |
| microsoft/phi-2        | 2.7B              | 1           | 0.43      |
| Qwen/Qwen1.5-0.5B-Chat | 0.5B              | 1           | 0.05      |

You can also find a leaderboard of the performance of different models on the HumanEval dataset [here](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard), and [here](https://qwenlm.github.io/blog/qwen1.5/) for `Qwen1.5` model. My results may differ from the ones reported in the leaderboard due to the different prompt engineering, checkout how Qwen is generating completions for the [normal](https://github.com/QwenLM/Qwen/blob/main/eval/evaluate_humaneval.py) and [chat](https://github.com/QwenLM/Qwen/blob/main/eval/evaluate_chat_humaneval.py) models.

Note: You can notice that the reported performance 

# Datasets

## Code Alpaca 20k

Preprocessing steps:
0. Initial dataset size is 20k samples.
1. Remove samples that no compile. Dataset size is 19,999 samples.
2. Remove samples that are not function definitions. Dataset size is 19,999 samples.
3. Remove samples that are not indentated with 4 spaces. Dataset size is 19,999 samples.

For implementation details, check the `get_code_alpaca_20k` function in the [datasets](src/utils/datasets.py) file.

# Evaluation: HumanEval

The evaluation of the performance of our systems is vital to know if the design choices we made are effective. We will use the [HumanEval framework](https://github.com/openai/human-eval/tree/master) to evaluate the quality of the generated code. This is a toy project and we only will use this dataset based on `python` programming languages, but more datasets can be added in the future. Check out the [APPS](https://github.com/hendrycks/apps), [MBPP](https://huggingface.co/datasets/mbpp), [MultiPL-E](https://github.com/nuprl/MultiPL-E), or [DS-1000](https://ds1000-code-gen.github.io/) benchmarks for more information.

HumanEval is a evaluation set we release to measure functional correctness for synthesizing programs from docstrings released by OpenAI. The dataset contains 164 problems and their corresponding unit test. Additionatlly, authors provide the necessary code to easily evaluate the performance of the model on this dataset. Check out [HumanEval repository](https://github.com/openai/human-eval/tree/master).

## Evaluation Execution

To evaluate the performance of a model on the HumanEval dataset, you need to follow the next steps:
1. Clone the HumanEval repository, install it following their instructions, and unzip the dataset located in the `data` folder. Each problem is stored in a JSON file with the following structure:
```json
{
    "task_id": "Task ID",
    "prompt": "Prompt",
    "entry_point": "Entry Point",
    "canonical_solution": "Canonical Solution",
    "test": "Unit Tests"
}
```
2. We will now need to generate completions for the problems in the dataset using the `prompt` field. To do this, we will use an LLM model. The completions should be saved in a JSON Lines (jsonl) format, where each sample is formatted into a single line like so:
```json
{"task_id": "Corresponding HumanEval task ID", "completion": "Completion only without the prompt"}
```
3. Once we have the completions, we can evaluate the performance of the model running the CLI script provided by the HumanEval repository. The script will output the performance of the model on the dataset. To run the script, you need to execute the following command:
```bash
evaluate_functional_correctness your_completions.jsonl
```

## Case Example

Let's see an example of what the prompt and completions look like for a problem in the HumanEval dataset. The problem is the following:
- Task ID: `HumanEval/2`
- Prompt: 
```python
def truncate_number(number: float) -> float:
    """ 
    Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).
    Return the decimal part of the number.
    >>> truncate_number(3.5)
        0.5
    """
```
- Entry Point: `truncate_number`
- Canonical Solution: 
```python
    return number % 1.0
```
- Test:
```python
METADATA = {
    'author': 'jt',
    'dataset': 'test'
}

def check(candidate):
    assert candidate(3.5) == 0.5
    assert abs(candidate(1.33) - 0.33) < 1e-6
    assert abs(candidate(123.456) - 0.456) < 1e-6
```


