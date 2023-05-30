import numpy as np
import torch

# This is the unfused version of StableAdamW. It is slower than the fused version (coming).


class StableAdamWUnfused(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0.002,
        weight_decay=0.2,
        betas=(0.9, 0.99),
        eps=1e-8,
        clip_thresh=1.0,
        precision="amp_bfloat16",
        custom_scalar=65536,
    ):
        beta1, beta2 = betas[0], betas[1]
        defaults = dict(lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2)
        super(StableAdamWUnfused, self).__init__(params, defaults)

        self.eps = eps
        self.d = clip_thresh

        # Set precision to "custom_fp16" if you want to use a fixed loss scalar, custom_scalar, which is divided out in the update step.
        # If you do this, call (custom_scalar * loss).backward() instead of loss.backward().
        self.precision = precision
        self.custom_scaler = custom_scalar

        for group in self.param_groups:
            group["step"] = 1.0

        print("Using StableAdamWUnfused-v1")

    def __setstate__(self, state):
        super(StableAdamWUnfused, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            step = group["step"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                theta = p.data
                param_state = self.state[p]

                if self.precision == "custom_fp16":
                    g = p.grad.data / self.custom_scaler
                    if torch.any(torch.isnan(g) | torch.isinf(g)):
                        continue
                else:
                    g = p.grad.data

                if "exp_avg" not in param_state:
                    v = param_state["exp_avg"] = torch.zeros_like(theta)
                    u = param_state["exp_avg_sq"] = torch.zeros_like(theta)
                else:
                    v = param_state["exp_avg"]
                    u = param_state["exp_avg_sq"]

                beta1hat = beta1 * (1 - beta1 ** (step - 1)) / (1 - beta1**step)
                beta2hat = beta2 * (1 - beta2 ** (step - 1)) / (1 - beta2**step)

                v = v.mul_(beta1hat).add_(g, alpha=1.0 - beta1hat)
                u = u.mul_(beta2hat).addcmul_(g, g, value=1.0 - beta2hat)

                denominator = u.sqrt().add_(self.eps)

                # StableAdamW = AdamW + update clipping (https://arxiv.org/abs/1804.04235) applied tensor-wise.
                rms = (
                    torch.div(
                        g.pow(2), torch.maximum(u, (self.eps**2) * torch.ones_like(u))
                    )
                    .mean()
                    .sqrt()
                    .item()
                )

                theta = theta.mul_(1.0 - lr * weight_decay).addcdiv_(
                    v, denominator, value=-lr * (1.0 / max(1.0, rms / self.d))
                )

                # save current params
                param_state["exp_avg"] = v
                param_state["exp_avg_sq"] = u

            group["step"] = step + 1
