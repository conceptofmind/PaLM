import math
import multiprocessing
import os
from datetime import timedelta
from functools import partial
from itertools import chain

import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from accelerate import Accelerator
from accelerate.utils import (DummyOptim, DummyScheduler,
                              InitProcessGroupKwargs)
from datasets import concatenate_datasets, load_dataset
from lion_pytorch import Lion
from palm_rlhf_pytorch import PaLM
from palm_rlhf_pytorch.palm import LayerNorm, ParallelTransformerBlock
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)

from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)


from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoTokenizer, default_data_collator,
                          get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup, set_seed)

from palm.stable_adamw import StableAdamWUnfused

# constants


class CFG:
    BATCH_SIZE: int = 3
    GRADIENT_ACCUMULATE_EVERY: int = 1
    SEED: int = 42
    LEARNING_RATE: float = 3e-4
    WEIGHT_DECAY: float = 0.1
    SEQ_LEN: int = 8192
    NUM_CPU: int = multiprocessing.cpu_count()
    USE_DEEPSPEED: bool = False
    USE_FSDP: bool = False
    USE_PRETOKENIZED: bool = True
    USE_ACTIVATION_CHECKPOINTING: bool = False
    RESUME_FROM_CHECKPOINT: str = None
    CHECKPOINTING_STEPS: int = 1000
    OUTPUT_DIR: str = "YOUR_OUTPUT_DIR"
    ENTITY_NAME: str = "YOUR_ENTITY_NAME"


# helpers


def print_num_params(model, accelerator: Accelerator):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Number of parameters in model: {n_params}")


# activation checkpointing


def activation_checkpointing(
    model: torch.nn.Module,
    offload_to_cpu: bool = False,
    accelerator: Accelerator = None,
):
    """
    Apply activation checkpointing to a model.

    Args:
        model (Module): The model to which to apply activation checkpointing.
        offload_to_cpu (bool, optional): Whether to offload the activations to CPU. Defaults to False.
        accelerator (Accelerator, optional): The Accelerate library accelerator. Defaults to None.
    """
    if accelerator is not None:
        accelerator.print(f"Using activation checkpointing")
    check_fn = lambda submodule: isinstance(submodule, ParallelTransformerBlock)
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        offload_to_cpu=offload_to_cpu,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )


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


# learning rate scheduler


def get_lr_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    num_warmup_steps: int,
    max_train_steps: int,
    grad_accumulate_every: int = 1,
    accelerator: Accelerator = None,
):
    """
    Get a learning rate scheduler with warmup.

    Args:
        optimizer (Optimizer): The optimizer for which to create the learning rate scheduler.
        scheduler_type (str): The type of learning rate scheduler to create, either "linear" or "cosine".
        num_warmup_steps (int): The number of warmup steps for the learning rate scheduler.
        max_train_steps (int): The maximum number of training steps.
        grad_accumulate_every (int, optional): The gradient accumulation factor. Defaults to 1.
        accelerator (Accelerator, optional): The Accelerate library accelerator. Defaults to None.

    Returns:
        The learning rate scheduler with warmup.

    Raises:
        ValueError: If scheduler_type is not "linear" or "cosine".
    """
    NUM_WARMUP_STEPS = num_warmup_steps
    GRADIENT_ACCUMULATE_EVERY = grad_accumulate_every
    if accelerator is not None:
        accelerator.print(f"Using {scheduler_type} lr scheduler")
    if scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=NUM_WARMUP_STEPS * GRADIENT_ACCUMULATE_EVERY,
            num_training_steps=max_train_steps * GRADIENT_ACCUMULATE_EVERY,
        )
    elif scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=NUM_WARMUP_STEPS * GRADIENT_ACCUMULATE_EVERY,
            num_training_steps=max_train_steps * GRADIENT_ACCUMULATE_EVERY,
        )
    else:
        raise ValueError(
            "Invalid scheduler_type. Expected 'linear' or 'cosine', got: {}".format(
                scheduler_type
            )
        )


# optimizers


def decoupled_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    beta_1: float,
    beta_2: float,
    optimizer_type: str,
    use_fsdp: bool = True,
    accelerator: Accelerator = None,
):
    """
    Decouples the optimizer from the training process.

    This function sets up the optimizer for the model by creating two groups of parameters:
    one for weight decay and one without weight decay. Then, it initializes the optimizer
    with these two groups of parameters.

    Args:
        model (Module): The model whose parameters are optimized.
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.
        beta_1 (float): The exponential decay rate for the 1st moment estimates.
        beta_2 (float): The exponential decay rate for the 2nd moment estimates.
        optimizer_type (str): The type of the optimizer. Can be 'lion', 'adamw', or 'stable_adamw'.
        use_fsdp (bool, optional): If True, the optimizer will work with fully sharded data parallelism. Defaults to True.
        accelerator (Accelerator, optional): The accelerator from HuggingFace's Accelerate library. Defaults to None.

    Returns:
        Optimizer: The initialized optimizer.

    Raises:
        ValueError: If the optimizer type is not 'lion', 'adamw' or 'stable_adamw'.
    """
    accelerator.print(f"Using {optimizer_type} optimizer")
    # Create an empty dictionary called param_dict to store the model's named parameters.
    param_dict = {}
    # Iterate over the model's named parameters and populate the param_dict with key-value pairs.
    for param_name, param in model.named_parameters():
        param_dict[param_name] = param

    # Separate the model's named modules into two groups: decay and no_decay.

    # Create an empty list to store the names of the LayerNorm and Embedding layer weights with no weight decay.
    no_decay = []

    if use_fsdp:
        exclude_module = "_fsdp_wrapped_module.token_emb"
    else:
        exclude_module = "token_emb"

    # Iterate through the named modules of the model.
    for module_name, module in model.named_modules():
        # Check if the current module is an instance of any of the desired types (LayerNorm or torch.nn.Embedding).
        for ndim in [LayerNorm, torch.nn.Embedding]:
            if isinstance(module, ndim):
                # If torch.nn.Embedding, append its name with a ".weight" suffix to the no_decay list.
                if module_name == exclude_module:
                    no_decay.append(f"{module_name}.weight")
                else:
                    # If the module is an instance of LayerNorm
                    no_decay.append(f"{module_name}.gamma")
                # Exit the inner loop since the desired module has been found.
                break

    # Create an empty list to store the names of the Linear layer weights with weight decay.
    decay = []

    # Iterate through the named modules of the model.
    for module_name, module in model.named_modules():
        # Check if the current module is an instance of the desired type (torch.nn.Linear).
        for ndim in [torch.nn.Linear]:
            if isinstance(module, ndim):
                # If the module is an instance of torch.nn.Linear, append its name with a ".weight" suffix to the decay list.
                decay.append(f"{module_name}.weight")
                # Exit the inner loop since the desired module has been found.
                break

    # Create two separate lists of model parameters: decay_param and no_decay_param.
    # The decay_param list contains the parameters that should have weight decay applied.
    # The no_decay_param list contains the parameters that should not have weight decay applied, excluding the 'to_logits.weight' parameter.

    # Create an empty list called decay_param to store the parameters with weight decay.
    decay_param = []

    if use_fsdp:
        exclude_param = "_fsdp_wrapped_module.to_logits.weight"
    else:
        exclude_param = "to_logits.weight"

    # Iterate over the decay list, which contains the names of the parameters with weight decay.
    for param in decay:
        # Check if the current parameter is not 'to_logits.weight'.
        # Append the corresponding parameter from param_dict to the decay_param list.

        if param != exclude_param:
            decay_param.append(param_dict[param])

    # Create an empty list called no_decay_param to store the parameters without weight decay.
    no_decay_param = []

    # Iterate over the no_decay list, which contains the names of the parameters without weight decay.
    for param in no_decay:
        # Append the corresponding parameter from param_dict to the no_decay_param list.
        no_decay_param.append(param_dict[param])

    # Create a list called grouped_params that contains two dictionaries.
    # The first dictionary has the decay_param list and the corresponding weight_decay value.
    # The second dictionary has the no_decay_param list and a weight_decay value of 0.0.
    grouped_params = [
        {"params": decay_param, "weight_decay": weight_decay},
        {"params": no_decay_param, "weight_decay": 0.0},
    ]

    # Create a variable called optimizer that stores an instance of the optimizer.
    if optimizer_type == "lion":
        optimizer = Lion(grouped_params, lr=learning_rate, betas=(beta_1, beta_2),)
    elif optimizer_type == "adamw":
        optimizer = AdamW(grouped_params, lr=learning_rate, betas=(beta_1, beta_2),)
    elif optimizer_type == "deepspeed":
        optimizer = DummyOptim(grouped_params, lr=learning_rate, betas=(beta_1, beta_2),)
    elif optimizer_type == "stable_adamw":
        optimizer = StableAdamWUnfused(
            grouped_params, lr=learning_rate, betas=(beta_1, beta_2),
        )
    else:
        raise ValueError(
            "Invalid optimizer_type. Expected 'lion', 'adamw', 'deepspeed' or 'stable_adamw', got: {}".format(
                optimizer_type
            )
        )

    # Return the optimizer.
    return optimizer


# dataloaders


def build_dataloaders():
    """
    Build data loaders for training.

    This function performs the following steps:
    1. Load the tokenizer from the pretrained "EleutherAI/gpt-neox-20b" model.
    2. Load the "openwebtext" dataset.
    3. Tokenize the dataset, adding the end-of-sentence token to each text.
    4. Process the tokenized dataset into chunks of a specified block size.

    Returns:
        Dataset: The processed dataset ready for training.
    """
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    dataset = load_dataset("openwebtext", split="train")

    tokenized_dataset = dataset.map(
        lambda example: tokenizer([t + tokenizer.eos_token for t in example["text"]]),
        batched=True,
        num_proc=CFG.NUM_CPU,
        remove_columns=["text"],
    )

    block_size = CFG.SEQ_LEN

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

    train_dataset = tokenized_dataset.map(
        group_texts, batched=True, num_proc=CFG.NUM_CPU,
    )

    return train_dataset


def build_pre_tokenized():
    d0 = load_dataset("conceptofmind/c4_0-to-20_neox_with_eos_8k", split="train")
    d1 = load_dataset("conceptofmind/c4_21-to-40_neox_with_eos_8k", split="train")
    d2 = load_dataset("conceptofmind/c4_41-to-60_neox_with_eos_8k", split="train")
    d3 = load_dataset("conceptofmind/c4_61-to-80_neox_with_eos_8k", split="train")
    d4 = load_dataset("conceptofmind/c4_81-to-100_neox_with_eos_8k", split="train")
    train_dataset = concatenate_datasets([d0, d1, d2, d3, d4])
    return train_dataset


# main


def main():
    # accelerator

    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))

    accelerator = Accelerator(
        gradient_accumulation_steps=CFG.GRADIENT_ACCUMULATE_EVERY,
        mixed_precision="bf16",
        log_with="wandb",
        kwargs_handlers=[timeout],
    )

    accelerator.init_trackers(
        project_name="palm",
        config={
            "batch_size": CFG.BATCH_SIZE,
            "gradient_accumulate_every": CFG.GRADIENT_ACCUMULATE_EVERY,
            "learning_rate": CFG.LEARNING_RATE,
            "seq_len": CFG.SEQ_LEN,
        },
        init_kwargs={"wandb": {"entity": CFG.ENTITY_NAME}},
    )

    accelerator.print(f"Total GPUS: {accelerator.num_processes}")

    # set seed

    set_seed(CFG.SEED)

    # instantiate palm

    # 410m
    # model = PaLM(
    #     num_tokens=50304, dim=1024, depth=24, dim_head=128, heads=8, flash_attn=True, #qk_rmsnorm = True,
    # ).to(accelerator.device)

    # 1B
    model = PaLM(
        num_tokens=50304,
        dim=2048,
        depth=16,
        dim_head=128,
        heads=8,
        flash_attn=True,
        qk_rmsnorm=False,
    )

    print_num_params(model, accelerator)

    if CFG.USE_FSDP:
        model = fsdp(
            model,
            mp="bf16",
            shard_strat="SHARD_GRAD"
        )

    if CFG.USE_ACTIVATION_CHECKPOINTING:
        activation_checkpointing(model, accelerator)

    model = accelerator.prepare(model)

    # dataloaders

    if CFG.USE_PRETOKENIZED:
        train_dataset = build_pre_tokenized()
    else:
        train_dataset = build_dataloaders()

    train_loader = DataLoader(
        train_dataset, batch_size=CFG.BATCH_SIZE, collate_fn=default_data_collator,
    )

    # optimizer

    optim = decoupled_optimizer(
        model=model,
        learning_rate=CFG.LEARNING_RATE, 
        weight_decay=CFG.WEIGHT_DECAY, 
        beta_1=0.90, 
        beta_2=0.95, 
        optimizer_type='adamw',  
        use_fsdp=True,
        accelerator=accelerator
    )

    # Determine number of training steps

    max_train_steps = math.ceil(len(train_loader) / CFG.GRADIENT_ACCUMULATE_EVERY)
    accelerator.print(f"Max train steps: {max_train_steps}")

    # lr scheduler

    NUM_WARMUP_STEPS = int(max_train_steps * 0.01)
    accelerator.print(f"Num warmup steps: {NUM_WARMUP_STEPS}")

    if CFG.USE_DEEPSPEED:
        lr_scheduler = DummyScheduler(
            optim, 
            total_num_steps=max_train_steps * accelerator.num_processes, 
            warmup_num_steps=NUM_WARMUP_STEPS
        )
    else:
        lr_scheduler = get_lr_scheduler_with_warmup(
            optimizer=optim,
            scheduler_type="cosine",
            num_warmup_steps=NUM_WARMUP_STEPS,
            max_train_steps=max_train_steps,
            grad_accumulate_every=CFG.GRADIENT_ACCUMULATE_EVERY,
        )

    # prepare

    optim, train_loader, lr_scheduler = accelerator.prepare(
        optim, train_loader, lr_scheduler
    )

    # checkpoint scheduler

    accelerator.register_for_checkpointing(lr_scheduler)

    # I do not know why Huggingface recommends recalculation of max_train_steps

    max_train_steps = math.ceil(len(train_loader) / CFG.GRADIENT_ACCUMULATE_EVERY)
    accelerator.print(f"Max train steps recalculated: {max_train_steps}")

    # Total batch size for logging

    total_batch_size = (
        CFG.BATCH_SIZE * accelerator.num_processes * CFG.GRADIENT_ACCUMULATE_EVERY
    )
    accelerator.print(f"Total batch size: {total_batch_size}")

    # resume training

    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0

    if CFG.RESUME_FROM_CHECKPOINT:
        if CFG.RESUME_FROM_CHECKPOINT is not None or CFG.RESUME_FROM_CHECKPOINT != "":
            accelerator.print(f"Resuming from checkpoint {CFG.RESUME_FROM_CHECKPOINT}")
            accelerator.load_state(CFG.RESUME_FROM_CHECKPOINT)
            path = os.path.basename(CFG.RESUME_FROM_CHECKPOINT)
        training_difference = os.path.splitext(path)[0]

        # need to multiply `gradient_accumulation_steps` to reflect real steps
        resume_step = (
            int(training_difference.replace("step_", ""))
            * CFG.GRADIENT_ACCUMULATE_EVERY
        )

    if CFG.RESUME_FROM_CHECKPOINT and resume_step is not None:
        train_loader = accelerator.skip_first_batches(train_loader, resume_step)
        completed_steps += resume_step
        progress_bar.update(resume_step)

    # training

    model.train()
    for step, batch in enumerate(train_loader):
        with accelerator.accumulate(model):
            inputs = batch["input_ids"].to(accelerator.device)
            loss = model(inputs, return_loss=True)
            accelerator.backward(loss)

            accelerator.log({"loss": loss.item()}, step=step)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optim.step()
            lr_scheduler.step()
            optim.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            completed_steps += 1

        if isinstance(CFG.CHECKPOINTING_STEPS, int):
            if completed_steps % CFG.CHECKPOINTING_STEPS == 0:
                output_dir = f"step_{completed_steps }"
                if CFG.OUTPUT_DIR is not None:
                    output_dir = os.path.join(CFG.OUTPUT_DIR, output_dir)
                accelerator.save_state(output_dir)

        if completed_steps >= max_train_steps:
            break

    # end training

    # accelerator.print(f"Training Finished")
    accelerator.end_training()

    # save final model

    # accelerator.print(f"Saving model to {CFG.OUTPUT_DIR}")
    if CFG.OUTPUT_DIR is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        with accelerator.main_process_first():
            accelerator.save(
                unwrapped_model.state_dict(), f"{CFG.OUTPUT_DIR}/final/final_model.pt"
            )


if __name__ == "__main__":
    main()
