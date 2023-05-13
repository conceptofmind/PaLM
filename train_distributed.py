import math
import multiprocessing
import os
from datetime import timedelta
from functools import partial
from itertools import chain

import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datasets import concatenate_datasets, load_dataset
from palm_rlhf_pytorch import PaLM
from palm_rlhf_pytorch.palm import LayerNorm, ParallelTransformerBlock
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoTokenizer, default_data_collator,
                          get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup, set_seed)

from stable_adamw import StableAdamWUnfused

# constants


class CFG:
    BATCH_SIZE: int = 3
    GRADIENT_ACCUMULATE_EVERY: int = 1
    SEED: int = 42
    LEARNING_RATE: float = 3e-4
    WEIGHT_DECAY: float = 0.1
    SEQ_LEN: int = 8192
    NUM_CPU: int = multiprocessing.cpu_count()
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


def fsdp_activation_checkpointing(
    model, accelerator: Accelerator, offload_to_cpu=False
):

    accelerator.print(f"Using FSDP activation checkpointing")

    check_fn = lambda submodule: isinstance(submodule, ParallelTransformerBlock)

    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        offload_to_cpu=offload_to_cpu,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )


def get_lr_scheduler_with_warmup(
    optimizer, scheduler_type, num_warmup_steps, max_train_steps, grad_accumulate_every
):
    NUM_WARMUP_STEPS = num_warmup_steps
    GRADIENT_ACCUMULATE_EVERY = grad_accumulate_every

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
    model, learning_rate, weight_decay, beta_1, beta_2, use_adamw=False
):
    # Create an empty dictionary called param_dict to store the model's named parameters.
    param_dict = {}
    # Iterate over the model's named parameters and populate the param_dict with key-value pairs.
    for param_name, param in model.named_parameters():
        param_dict[param_name] = param

    # Separate the model's named modules into two groups: decay and no_decay.

    # Create an empty list to store the names of the LayerNorm and Embedding layer weights with no weight decay.
    no_decay = []

    # Iterate through the named modules of the model.
    for module_name, module in model.named_modules():
        # Check if the current module is an instance of any of the desired types (LayerNorm or torch.nn.Embedding).
        for ndim in [LayerNorm, torch.nn.Embedding]:
            if isinstance(module, ndim):
                # If torch.nn.Embedding, append its name with a ".weight" suffix to the no_decay list.
                if module_name == "token_emb":
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

    # Iterate over the decay list, which contains the names of the parameters with weight decay.
    for param in decay:
        # Check if the current parameter is not 'to_logits.weight'.
        # Append the corresponding parameter from param_dict to the decay_param list.
        if param != "to_logits.weight":
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
    if use_adamw:
        optimizer = AdamW(grouped_params, lr=learning_rate, betas=(beta_1, beta_2),)
    else:
        optimizer = StableAdamWUnfused(
            grouped_params, lr=learning_rate, betas=(beta_1, beta_2),
        )

    # Return the optimizer.
    return optimizer


# dataloaders


def build_dataloaders():
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
        flash_attn=True,  # qk_rmsnorm = True,
    ).to(accelerator.device)

    print_num_params(model, accelerator)

    if CFG.USE_ACTIVATION_CHECKPOINTING:
        fsdp_activation_checkpointing(model, accelerator)

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
        model,
        learning_rate=CFG.LEARNING_RATE,
        weight_decay=CFG.WEIGHT_DECAY,
        beta_1=0.9,
        beta_2=0.95,
        use_adamw=False,
    )

    # Determine number of training steps

    max_train_steps = math.ceil(len(train_loader) / CFG.GRADIENT_ACCUMULATE_EVERY)
    accelerator.print(f"Max train steps: {max_train_steps}")

    # lr scheduler
    # We cant decide on an actual number

    NUM_WARMUP_STEPS = int(max_train_steps * 0.01)
    accelerator.print(f"Num warmup steps: {NUM_WARMUP_STEPS}")

    lr_scheduler = get_lr_scheduler_with_warmup(
        optimizer=optim,
        scheduler_type="cosine",
        num_warmup_steps=NUM_WARMUP_STEPS,
        max_train_steps=max_train_steps,
        grad_accumulate_every=CFG.GRADIENT_ACCUMULATE_EVERY,
    )

    # prepare

    model, optim, train_loader, lr_scheduler = accelerator.prepare(
        model, optim, train_loader, lr_scheduler
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
                accelerator.clip_grad_norm_(model.parameters(), 0.5)

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
