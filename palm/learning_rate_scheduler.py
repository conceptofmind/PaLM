import math

import torch
from torch.optim.lr_scheduler import _LRScheduler

from palm.utils import print_main

# Cosine schedule with warmup

class WarmupCosineSchedule(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(WarmupCosineSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        else:
            progress = float(self.last_epoch - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            return [base_lr * (0.5 * (1.0 + math.cos(math.pi * progress))) / 10 for base_lr in self.base_lrs]

# Linear schedule with warmup

class WarmupLinearSchedule(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(WarmupLinearSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        else:
            progress = float(self.last_epoch - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            return [base_lr * (1.0 - progress) for base_lr in self.base_lrs]

# learning rate scheduler

def get_lr_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    num_warmup_steps: int,
    max_train_steps: int,
    grad_accumulate_every: int = 1,
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

    print_main(f"Using {scheduler_type} lr scheduler")
    if scheduler_type == "linear":
        return WarmupLinearSchedule(
            optimizer=optimizer,
            warmup_steps=NUM_WARMUP_STEPS * GRADIENT_ACCUMULATE_EVERY,
            total_steps=max_train_steps * GRADIENT_ACCUMULATE_EVERY,
        )
    elif scheduler_type == "cosine":
        return WarmupCosineSchedule(
            optimizer=optimizer,
            warmup_steps=NUM_WARMUP_STEPS * GRADIENT_ACCUMULATE_EVERY,
            total_steps=max_train_steps * GRADIENT_ACCUMULATE_EVERY,
        )
    else:
        raise ValueError(
            "Invalid scheduler_type. Expected 'linear' or 'cosine', got: {}".format(
                scheduler_type
            )
        )
