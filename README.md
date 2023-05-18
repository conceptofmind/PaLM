# PaLM
<img src="./palm.gif" width="450px"></img>

## Acknowledgements
- <a href="https://github.com/CarperAI">CarperAI</a>, <a href="https://twitter.com/lcastricato">Louis Castricato</a>, and <a href="https://stability.ai/">Stability.ai</a> for the very generous sponsorship to work on machine learning research.
- <a href="https://github.com/lucidrains">Phil Wang (Lucidrains)</a> for his inspiring work and input on training and architectures.
- <a href="https://twitter.com/dmayhem93">Dakota ("He berk reacted once")</a>, <a href="https://twitter.com/jonbtow">Guac</a>, <a href="https://twitter.com/zach_nussbaum">Zach</a>, and <a href="https://twitter.com/aman_gif">Aman</a> for providing information about Huggingface and Slurm. I typically only use Apex and DeepSpeed.

## FAQ
Three different size PaLM models (150m, 410m, 1b) have been trained with 8k context length on all of <a href="https://huggingface.co/datasets/c4">C4</a>. The models are compatible with Lucidrain's <a href="https://github.com/lucidrains/toolformer-pytorch">Toolformer-pytorch</a>, <a href="https://github.com/lucidrains/PaLM-pytorch">PaLM-pytorch</a>, and <a href="https://github.com/lucidrains/PaLM-rlhf-pytorch">PaLM-rlhf-pytorch</a>. A fourth 2b model is currently being trained. These are currently the baseline versions of the models and additional training will be done at a larger scale. All of the models will be further instruction-tuned on FLAN to provide flan-PaLM models.

The models were trained with <a href="https://github.com/HazyResearch/flash-attention">Flash Attention</a>, <a href="https://arxiv.org/abs/2212.10554">Xpos Rotary Embeddings</a> for better length extrapolation, and <a href="https://arxiv.org/abs/1911.02150">multi-query single-key-value attention</a> for more efficient decoding. The models have been uploaded to Torch hub and the files are additionally stored on the Huggingface hub. You can find the model each of the PyTorch model files here: <a href="https://huggingface.co/conceptofmind/palm-150m">PaLM-150m</a>, <a href="https://huggingface.co/conceptofmind/palm-410m">PaLM-410m</a>, <a href="https://huggingface.co/conceptofmind/palm-1b">PaLM-1b</a>. If the models are not downloading from Torch hub correctly be sure to clear out the checkpoint and model folders in `.cache/torch/hub/`. If that still does not resolve the issue then you can download the files from the Huggingface repositories. Huggingface integration is currently a work-in-progress.

All of the training data has been pre-tokenized with the GPTNEOX tokenizer and blocked at sequence lengths of 8192. This will help to save the large cost of preprocessing data. The datasets are available on Huggingface in parquet format and chunks here: <a href="https://huggingface.co/datasets/conceptofmind/c4_0-to-20_neox_with_eos_8k">C4 Chunk 1</a>, <a href="https://huggingface.co/datasets/conceptofmind/c4_21-to-40_neox_with_eos_8k">C4 Chunk 2</a>, <a href="https://huggingface.co/datasets/conceptofmind/c4_41-to-60_neox_with_eos_8k">C4 Chunk 3</a>, <a href="https://huggingface.co/datasets/conceptofmind/c4_61-to-80_neox_with_eos_8k">C4 Chunk 4</a>, and <a href="https://huggingface.co/datasets/conceptofmind/c4_81-to-100_neox_with_eos_8k">C4 Chunk 5</a>. There is also another option in the distributed training script to not used the provided pre-tokenized C4 dataset and instead load and process another dataset such as openwebtext.

## Installation
Make sure you install the requirements before trying to run the models.
```bash
git clone https://github.com/conceptofmind/PaLM.git
cd PaLM/
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
    num_tokens=50304, dim=1024, depth=24, dim_head=128, heads=8, flash_attn=True, qk_rmsnorm = False,
).cuda()

model.load('/palm_410m_8k_v0.pt')
```
If you would like to use the models on CPU you can do:
```python
device = torch.device("cpu")

model = PaLM(
    num_tokens=50304, dim=1024, depth=24, dim_head=128, heads=8, flash_attn=False, qk_rmsnorm = False,
).to(device).eval()

checkpoint = torch.load('./palm_410m_8k_v0.pt', map_location=device)
model.load_state_dict(checkpoint)
```
To generate text with the models you can use the command line:
- prompt - Text prompt to generate text.
- seq_len - Sequence length for generated text. Default is 256.
- temperature - Sampling temperature. Default is 0.8.
- filter_thres - Filter threshold for sampling. Default is 0.9.
- dtype - A flag that can be used to change the dtype of the model for inference with Flash Attention. Requires an A100 GPU.
- model - Model to use for generation. There are three different model sizes (150m, 410m, 1b): "palm_150m_8k_v0", "palm_410m_8k_v0", and "palm_1b_8k_v0". Default is "palm_410m_8k_v0".

```bash
python3 inference.py "My dog is very cute" --seq_len 256 --temperature 0.8 --filter_thres 0.9 --model "palm_410m_8k_v0"
```
Inference uses `torch.compile()`, <a href="https://github.com/HazyResearch/flash-attention">Flash Attention</a>, and <a href="https://pytorch.org/blog/introducing-hidet/">Hidet</a> for performance improvements. A generic inference script, `inference.py` is provided for you to play around with if you want to extend generation by adding streaming or other features.

An example generation with the 410 million parameter model is:
>My dog is very cute, but not very good at socializing with other dogs. The dog loves all new people and he likes to hang out with other dogs. I do need to take him to the park with other dogs. He does have some bad puppy breath, but it is only when he runs off in a direction he doesn't want to go.
>currently my dog is being very naughty. He would like to say hi in the park, but would rather take great care of himself for a while. He also has bad breath. I am going to have to get him some oral braces. It's been 3 months.
>The dog has some biting pains around his mouth. The dog is very timid and scared. The dog gets aggressive towards people.
>The dog is very playful and they are a little spoiled. I am not sure if it's a dog thing or if he is spoiled. He loves his toys and just wants to play. He plays with his toys all the time and even goes on walks. He is a little picky, not very good with other dogs.
>The dog is just a little puppy that goes to the park. He is a super friendly dog. He has not had a bad mouth or bad breath
An example generation with the 1 billion parameter models:
>I was just thinking we’d have a super fun family engagement session outside the city but the air was pretty gusty and the rain was coming down and the sun was setting so we decided we should be outside.
>We started out at the NC Arboretum in Chapel Hill and it was such a great place for family photos. You can’t beat the views here by the rose gardens and the light is stunning at night.
>We got to Spencer’s dad’s house in Chapel Hill and got to play with his kids for a bit and then set up for a photoshoot

## Training
I provide a distributed training script, `train_distributed.py` which was used to train each of the models. I used accelerate and slurm for multinode training. The models were trained on 64 A100 (80 GB) GPUs. You can freely change the model layers and hyperparameter configuration to meet your hardware requirements. The models were trained with <a href="https://github.com/HazyResearch/flash-attention">Flash Attention</a>, <a href="https://arxiv.org/abs/2212.10554">Xpos Rotary Embeddings</a> for better length extrapolation, and <a href="https://arxiv.org/abs/1911.02150">multi-query single-key-value attention</a> for more efficient decoding. I used decoupled weight decay Adam W for training. There is the option to use <a href="https://twitter.com/Mitchnw">Mitchell Wortsman's</a> Stable Adam W as well. I will be testing this for larger runs. You are able to load the models weights and alter the training script to fine-tune the models. I will be adding a specific fine-tuning script in the very near future with exploration into LoRA.

| Model Size | Num Tokens | Dim | Depth | Dim Head | Heads | Flash Attention | Learning Rate |
| -------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| 150 M | 50304 | 768 | 12 | 128 | 8 | True | 6e-4 |
| 410 M | 50304 | 1024 | 24 | 128 | 8 | True | 3e-4 |
| 1 B | 50304 | 2048 | 16 | 128 | 8 | True | 3e-4 |

## Data
You can preprocess a different dataset in a way similar to the C4 dataset used during training by running the `build_dataset.py` script. This will pre-tokenize, chunk the data in blocks of a specified sequence length, and upload to the Huggingface hub. For example:
```bash
python3 build_dataset.py --seed 42 --seq_len 8192 --hf_account "your_hf_account" --tokenizer "EleutherAI/gpt-neox-20b" --dataset_name "EleutherAI/the_pile_deduplicated"
```

## Experiments

I tried numerous different experiments related QK normalization, XPos, Stable Adam W, and long context training. I will be releasing logs and findings.

## Citations 

```bibtex
@inproceedings{Chowdhery2022PaLMSL,
    title   = {PaLM: Scaling Language Modeling with Pathways},
    author  = {Aakanksha Chowdhery and Sharan Narang and Jacob Devlin and Maarten Bosma and Gaurav Mishra and Adam Roberts and Paul Barham and Hyung Won Chung and Charles Sutton and Sebastian Gehrmann and Parker Schuh and Kensen Shi and Sasha Tsvyashchenko and Joshua Maynez and Abhishek Rao and Parker Barnes and Yi Tay and Noam M. Shazeer and Vinodkumar Prabhakaran and Emily Reif and Nan Du and Benton C. Hutchinson and Reiner Pope and James Bradbury and Jacob Austin and Michael Isard and Guy Gur-Ari and Pengcheng Yin and Toju Duke and Anselm Levskaya and Sanjay Ghemawat and Sunipa Dev and Henryk Michalewski and Xavier Garc{\'i}a and Vedant Misra and Kevin Robinson and Liam Fedus and Denny Zhou and Daphne Ippolito and David Luan and Hyeontaek Lim and Barret Zoph and Alexander Spiridonov and Ryan Sepassi and David Dohan and Shivani Agrawal and Mark Omernick and Andrew M. Dai and Thanumalayan Sankaranarayana Pillai and Marie Pellat and Aitor Lewkowycz and Erica Oliveira Moreira and Rewon Child and Oleksandr Polozov and Katherine Lee and Zongwei Zhou and Xuezhi Wang and Brennan Saeta and Mark Diaz and Orhan Firat and Michele Catasta and Jason Wei and Kathleen S. Meier-Hellstern and Douglas Eck and Jeff Dean and Slav Petrov and Noah Fiedel},
    year    = {2022}
}
```
```bibtex
@inproceedings{Sun2022ALT,
    title     = {A Length-Extrapolatable Transformer},
    author    = {Yutao Sun and Li Dong and Barun Patra and Shuming Ma and Shaohan Huang and Alon Benhaim and Vishrav Chaudhary and Xia Song and Furu Wei},
    year      = {2022}
}
```
```bibtex
@inproceedings{dao2022flashattention,
    title   = {Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
    author  = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
    booktitle = {Advances in Neural Information Processing Systems},
    year    = {2022}
}
```
```bibtex
@misc{zhao2023pytorch,
    title={PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel}, 
    author={Yanli Zhao and Andrew Gu and Rohan Varma and Liang Luo and Chien-Chin Huang and Min Xu and Less Wright and Hamid Shojanazeri and Myle Ott and Sam Shleifer and Alban Desmaison and Can Balioglu and Bernard Nguyen and Geeta Chauhan and Yuchen Hao and Shen Li},
    year={2023},
}
```