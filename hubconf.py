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

    hf_url = 'https://huggingface.co/conceptofmind/palm-150m/resolve/main/palm_150m_8k_v0.pt'
    state_dict = torch.hub.load_state_dict_from_url(hf_url)
    model.load_state_dict(state_dict)

    return model

def palm_410m_8k_v0():
    num_tokens = 50304
    dim = 1024
    depth = 24
    dim_head = 128
    heads = 8
    flash_attn = True

    model = PaLM(
        num_tokens=num_tokens, dim=dim, depth=depth, dim_head=dim_head, heads=heads, flash_attn=flash_attn
    )

    hf_url = 'https://huggingface.co/conceptofmind/palm-410m/resolve/main/palm_410m_8k_v0.pt'
    state_dict = torch.hub.load_state_dict_from_url(hf_url)
    model.load_state_dict(state_dict)

    return model

def palm_1b_8k_v0():
    num_tokens = 50304
    dim = 2048
    depth = 16
    dim_head = 128
    heads = 8
    flash_attn = True

    model = PaLM(
        num_tokens=num_tokens, dim=dim, depth=depth, dim_head=dim_head, heads=heads, flash_attn=flash_attn
    )

    hf_url = 'https://huggingface.co/conceptofmind/palm-1b/resolve/main/palm_1b_8k_v0.pt'
    state_dict = torch.hub.load_state_dict_from_url(hf_url)
    model.load_state_dict(state_dict)

    return model