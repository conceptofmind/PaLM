from functools import partial

import torch
from palm_rlhf_pytorch.palm import ParallelTransformerBlock
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)

from palm.utils import print_main

# activation checkpointing


def activation_checkpointing(
    model: torch.nn.Module,
    offload_to_cpu: bool = False,
):
    """
    Apply activation checkpointing to a model.

    Args:
        model (Module): The model to which to apply activation checkpointing.
        offload_to_cpu (bool, optional): Whether to offload the activations to CPU. Defaults to False.
        accelerator (Accelerator, optional): The Accelerate library accelerator. Defaults to None.
    """
    print_main(f"Using activation checkpointing")
    check_fn = lambda submodule: isinstance(submodule, ParallelTransformerBlock)
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        offload_to_cpu=offload_to_cpu,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )
