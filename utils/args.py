import argparse


class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


parser = argparse.ArgumentParser(description='Cody - Finetuning for Code Generation', formatter_class=SmartFormatter)

# google/gemma-2b | google/gemma-2b-it | microsoft/phi-2
# Qwen/Qwen1.5-0.5B | Qwen/Qwen1.5-0.5B-Chat
parser.add_argument('--model_name', type=str, default='Qwen/Qwen1.5-0.5B-Chat', help='Model name to use')

parser.add_argument('--epochs', type=int, default=10, help='Total number epochs for finetuning')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size for training')
parser.add_argument('--desired_batch_size', type=int, default=8, help='Final batch size after gradient accumulation')

parser.add_argument('--test_size', type=float, default=0.1, help='Data fraction for test')
parser.add_argument('--eval_interval', type=int, default=25, help='Interval for evaluation')

parser.add_argument('--lora_rank', type=int, default=32)
parser.add_argument('--lora_alpha', type=float, default=32, help='Learning rate')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='Learning rate')

parser.add_argument('--exp_id', type=str, help='Experiment ID')

args = parser.parse_args()

assert args.desired_batch_size > args.batch_size, "Desired batch size should be greater than batch size"
assert args.desired_batch_size % args.batch_size == 0, "Batch size should be divisible by desired batch size"
