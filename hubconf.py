#dependencies = ['torch', 'palm-rlhf-pytorch', 'accelerate', 'beartype', 'einops', 'lion-pytorch', 'tqdm']

import torch
from palm_rlhf_pytorch import PaLM

def palm_150m_8k_v0():
    num_tokens = 50304
    dim = 768
    depth = 12
    dim_head = 128
    heads = 8
    flash_attn = True

    model = PaLM(
        num_tokens=num_tokens, dim=dim, depth=depth, dim_head=dim_head, heads=heads, flash_attn=flash_attn
    )

    huggingface_url = 'https://huggingface.co/conceptofmind/palm-150m/resolve/main/palm_150m_8k_v0.pt'
    state_dict = torch.hub.load_state_dict_from_url(huggingface_url)
    model.load_state_dict(state_dict)

    return model
