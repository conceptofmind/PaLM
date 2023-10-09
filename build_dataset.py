import multiprocessing
import argparse
from itertools import chain
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed

def main(args):
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    train_dataset = load_dataset(args.dataset_name, split="train", data_dir=args.data_dir)

    def tokenize_function(example):
        return tokenizer([t + tokenizer.eos_token for t in example["text"]])

    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_workers,
        remove_columns=["text"],
    )

    block_size = args.seq_len

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    train_tokenized_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=args.num_workers,
    )

    train_tokenized_dataset.push_to_hub(args.hf_account_repo)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process and push dataset to Hugging Face Hub")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--seq_len", type=int, default=8192, help="Sequence length for processing")
    parser.add_argument("--hf_account_repo", type=str, default="YOUR HF ACCOUNT/REPO NAME", help="Hugging Face account name and repo")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b", help="Tokenizer model to use")
    parser.add_argument("--dataset_name", type=str, default="EleutherAI/the_pile_deduplicated", help="Name of the dataset to process")
    parser.add_argument("--data_dir", type=str, default=None, help="Name of the dataset directory to process")
    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count(), help="Number of workers for processing the data")
    args = parser.parse_args()
    main(args)
