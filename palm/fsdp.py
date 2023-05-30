from functools import partial

import torch
from palm_rlhf_pytorch.palm import ParallelTransformerBlock
from torch.distributed.fsdp import (BackwardPrefetch, FullyShardedDataParallel,
                                    MixedPrecision, ShardingStrategy)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# FSDP

def fsdp(
    model: torch.nn.Module,
    auto_wrap: bool = False,
    mp: str = "fp32",
    shard_strat: str = "NO_SHARD",
):
    """
    This function wraps a given PyTorch model with the FullyShardedDataParallel (FSDP) wrapper to enable efficient data parallelism and model sharding.

    Args:
        model (torch.nn.Module): The original PyTorch model to be wrapped with FSDP.
        auto_wrap (bool, optional): If True, it enables automatic wrapping of the model's layers according to the transformer_auto_wrap_policy. Default is False.
        mp (str, optional): The mixed precision mode to be used. Can be 'bf16' for BFloat16, 'fp16' for Float16 or 'fp32' for Float32 precision. Default is 'fp32'.
        shard_strat (str, optional): The sharding strategy to be used. Can be 'SHARD_GRAD' for sharding at gradient computation, 'FULL_SHARD' for full model sharding or 'NO_SHARD' for no sharding. Default is 'NO_SHARD'.

    Raises:
        ValueError: If the provided mp (mixed precision mode) is not 'bf16', 'fp16' or 'fp32'.
        ValueError: If the provided shard_strat (sharding strategy) is not 'SHARD_GRAD', 'FULL_SHARD' or 'NO_SHARD'.

    Returns:
        torch.nn.Module: The input model wrapped with FSDP.
    """
    if auto_wrap:
        palm_auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                ParallelTransformerBlock,
            },
        )
    else:
        palm_auto_wrap_policy = None

    if mp == "bf16":
        mp_fsdp = MixedPrecision(
            param_dtype=torch.bfloat16,
            # Gradient communication precision.
            reduce_dtype=torch.bfloat16,
            # Buffer precision.
            buffer_dtype=torch.bfloat16,
        )
    elif mp == "fp16":
        mp_fsdp = MixedPrecision(
            param_dtype=torch.float16,
            # Gradient communication precision.
            reduce_dtype=torch.float16,
            # Buffer precision.
            buffer_dtype=torch.float16,
        )
    elif mp == "fp32":
        mp_fsdp = MixedPrecision(
            param_dtype=torch.float32,
            # Gradient communication precision.
            reduce_dtype=torch.float32,
            # Buffer precision.
            buffer_dtype=torch.float32,
        )
    else:
        raise ValueError(
            "Invalid scheduler_type. Expected 'bf16', 'fp16' or 'fp32', got: {}".format(
                mp
            )
        )

    if shard_strat == "SHARD_GRAD":
        sharding_strat_fsdp = ShardingStrategy.SHARD_GRAD_OP 
    elif shard_strat == "FULL_SHARD":
        sharding_strat_fsdp = ShardingStrategy.FULL_SHARD
    elif shard_strat == "NO_SHARD":
        sharding_strat_fsdp = ShardingStrategy.NO_SHARD
    else:
        raise ValueError(
            "Invalid scheduler_type. Expected 'SHARD_GRAD', 'FULL_SHARD' or 'NO_SHARD', got: {}".format(
                shard_strat
            )
        )

    model = FullyShardedDataParallel(
        model,
        auto_wrap_policy=palm_auto_wrap_policy,
        mixed_precision=mp_fsdp,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        sharding_strategy=sharding_strat_fsdp,
        forward_prefetch=True,
        use_orig_params=True,
    )

    return model
