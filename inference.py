import torch
import hidet
from transformers import AutoTokenizer
from einops._torch_specific import allow_ops_in_compiled_graph
import argparse


def main():
    allow_ops_in_compiled_graph()

    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

    parser = argparse.ArgumentParser(description="Generate text using PaLM model")
    parser.add_argument("prompt", type=str, help="Text prompt to generate text")
    parser.add_argument(
        "--seq_len", type=int, default=256, help="Sequence length for generated text"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument(
        "--filter_thres", type=float, default=0.9, help="Filter threshold for sampling"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="palm_1b_8k_v0",
        help="Model to use for generation",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        help="Data type for the model: 'bf16', or 'fp32'",
    )

    args = parser.parse_args()


    dtype = torch.float32
    if args.dtype == 'bf16':
        dtype = torch.bfloat16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.hub.load("conceptofmind/PaLM", args.model).to(device).to(dtype).eval()

    hidet.torch.dynamo_config.use_tensor_core(True)
    hidet.torch.dynamo_config.search_space(2) 

    opt_model = torch.compile(model, backend="hidet")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    encoded_text = tokenizer(args.prompt, return_tensors="pt")

    output_tensor = opt_model.generate(
        seq_len=args.seq_len,
        prompt=encoded_text["input_ids"].to(device),
        temperature=args.temperature,
        filter_thres=args.filter_thres,
        pad_value=0.0,
        eos_token=tokenizer.eos_token_id,
        return_seq_without_prompt=False,
        use_tqdm=True,
    )

    decoded_output = tokenizer.batch_decode(output_tensor, skip_special_tokens=True)

    return decoded_output


if __name__ == "__main__":
    generated_text = main()
    for text in generated_text:
        print(f"{text}")
