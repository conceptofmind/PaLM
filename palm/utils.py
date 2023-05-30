import torch.distributed as dist  # Add this line

def print_num_params(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if dist.is_available():
        if dist.get_rank() == 0:
            print(f"Number of parameters in model: {n_params}")
    else:
        print(f"Number of parameters in model: {n_params}")

def print_main(msg):
    if dist.is_available():
        if dist.get_rank() == 0:
            print(msg)
    else:
        print(msg)