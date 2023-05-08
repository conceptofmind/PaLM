# PaLM

## Acknowledgements
- <a href="https://github.com/CarperAI">CarperAI</a>, <a href="https://twitter.com/lcastricato">Louis Castricato</a>, and <a href="https://stability.ai/">Stability.ai</a> for the very generous sponsorship to work on machine learning research.
- <a href="https://github.com/lucidrains">Phil Wang (Lucidrains)</a> for his inspiring work and input on training and architectures.
- <a href="https://twitter.com/dmayhem93">Dakota</a>, <a href="https://twitter.com/jonbtow">Guac</a>, <a href="https://twitter.com/zach_nussbaum">Zach</a>, and <a href="">Aman</a> for answering my questions about Huggingface and Slurm.

## Installation
Make sure you install the requirements before trying to run the models.
```bash
pip3 install -r requirements.txt
```

## Usage
To generate text with the models you can use the command line:
- prompt - Text prompt to generate text
- seq_len" default=256, help="Sequence length for generated text"
- temperature", type=float, default=0.8, help="Sampling temperature"
- filter_thres", type=float, default=0.9, help="Filter threshold for sampling"
- model",
        type=str,
        default="palm_410m_8k_v0",
        help="Model to use for generation",
    )

```
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

## Experiments

##



