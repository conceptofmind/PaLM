# PaLM
<a href=""></a>

## Acknowledgements
- <a href="https://github.com/CarperAI">CarperAI</a>, <a href="https://twitter.com/lcastricato">Louis Castricato</a>, and <a href="https://stability.ai/">Stability.ai</a> for the very generous sponsorship to work on machine learning research.
- <a href="https://github.com/lucidrains">Phil Wang (Lucidrains)</a> for his inspiring work and input on training and architectures.
- <a href="https://twitter.com/dmayhem93">Dakota</a>, <a href="https://twitter.com/jonbtow">Guac</a>, <a href="https://twitter.com/zach_nussbaum">Zach</a>, and <a href="">Aman</a> for answering my questions about Huggingface and Slurm.



## FAQ
Three different size PaLM models (150m, 410m, 1b) have been trained with 8k context length on all of <a href="https://huggingface.co/datasets/c4">C4</a>. 

The models were trained with <a href="https://github.com/HazyResearch/flash-attention">Flash Attention</a> and <a href="https://arxiv.org/abs/2212.10554">Xpos Rotary Embeddings</a> for better length extrapolation. The models have been uploaded to Torch hub and the files are additionally stored on the Huggingface hub. You can find the model each of the PyTorch model files here: <a href="https://huggingface.co/conceptofmind/palm-150m">PaLM-150m</a>, <a href="https://huggingface.co/conceptofmind/palm-410m">PaLM-410m</a>, <a href="https://huggingface.co/conceptofmind/palm-1b">PaLM-1b</a>. If the models are not downloading from Torch hub correctly be sure to clear out the checkpoint and model folders in `.cache/torch/hub/`. If that still does not resolve the issue then you can download the files from the Huggingface repositories. 

All of the training data has been pre-tokenized with the GPTNEOX tokenizer and blocked at sequence lengths of 8192. This will help to save the large cost of preprocessing data. The datasets are available on Huggingface in parquet format and chunks here: <a href="https://huggingface.co/datasets/conceptofmind/c4_0-to-20_neox_with_eos_8k">C4 Chunk 1</a>, <a href="https://huggingface.co/datasets/conceptofmind/c4_21-to-40_neox_with_eos_8k">C4 Chunk 2</a>, <a href="https://huggingface.co/datasets/conceptofmind/c4_41-to-60_neox_with_eos_8k">C4 Chunk 3</a>, <a href="https://huggingface.co/datasets/conceptofmind/c4_61-to-80_neox_with_eos_8k">C4 Chunk 4</a>, and <a href="https://huggingface.co/datasets/conceptofmind/c4_81-to-100_neox_with_eos_8k">C4 Chunk 5</a>. This is also another option in the distributed training script to not used the provided pre-tokenized C4 dataset and instead load and process another dataset such as openwebtext.

## Installation
Make sure you install the requirements before trying to run the models.
```bash
pip3 install -r requirements.txt
```

## Usage
You can load the pretrained models for additional training or fine-tuning from Torch hub by using:
```python
model = torch.hub.load("conceptofmind/PaLM", "palm_410m_8k_v0").cuda()
```
You can also load the PyTorch model checkpoints directly by:
```python
from palm_rlhf_pytorch import PaLM

model = PaLM(
    num_tokens=50304, dim=768, depth=12, dim_head=128, heads=8, flash_attn=True, #qk_rmsnorm = True,
).cuda()

model.load('/palm_410m_8k_v0.pt')
```
To generate text with the models you can use the command line:
- prompt - Text prompt to generate text.
- seq_len - Sequence length for generated text. Default is 256.
- temperature - Sampling temperature. Default is 0.8.
- filter_thres - Filter threshold for sampling. Default is 0.9.
- model - Model to use for generation. There are three different model sizes (150m, 410m, 1b): "palm_150m_8k_v0", "palm_410m_8k_v0", and "palm_1b_8k_v0". Default is "palm_410m_8k_v0".

```bash
python3 inference.py "My dog is very cute" --seq_len 256 --temperature 0.8 --filter_thres 0.9 --model "palm_410m_8k_v0"
```
A generic inference script, `inference.py` is provided for you to play around with.

An example generation with the 410 million parameter model is:
>My dog is very cute, but not very good at socializing with other dogs. The dog loves all new people and he likes to hang out with other dogs. I do need to take him to the park with other dogs. He does have some bad puppy breath, but it is only when he runs off in a direction he doesn't want to go.
>currently my dog is being very naughty. He would like to say hi in the park, but would rather take great care of himself for a while. He also has bad breath. I am going to have to get him some oral braces. It's been 3 months.
>The dog has some biting pains around his mouth. The dog is very timid and scared. The dog gets aggressive towards people.
>The dog is very playful and they are a little spoiled. I am not sure if it's a dog thing or if he is spoiled. He loves his toys and just wants to play. He plays with his toys all the time and even goes on walks. He is a little picky, not very good with other dogs.
>The dog is just a little puppy that goes to the park. He is a super friendly dog. He has not had a bad mouth or bad breath






## Training

## Data

## Experiments

##



