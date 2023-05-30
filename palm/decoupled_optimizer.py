import torch
from palm_rlhf_pytorch.palm import LayerNorm
from torch.optim import AdamW

from palm.utils import print_main
from stable_adamw import StableAdamWUnfused

# optimizers


def decoupled_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float = 0.1,
    beta_1: float = 0.90,
    beta_2: float = 0.95,
    optimizer_type: str = "adamw",
    use_fsdp: bool = True,
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
    print_main(f"Using {optimizer_type} optimizer")
    # Create an empty dictionary called param_dict to store the model's named parameters.
    param_dict = {}
    # Iterate over the model's named parameters and populate the param_dict with key-value pairs.
    for param_name, param in model.named_parameters():
        print_main(param_name)
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
    if optimizer_type == "adamw":
        optimizer = AdamW(
            grouped_params, 
            lr=learning_rate, 
            betas=(beta_1, beta_2),
        )
    elif optimizer_type == "stable_adamw":
        optimizer = StableAdamWUnfused(
            grouped_params, 
            lr=learning_rate, 
            betas=(beta_1, beta_2),
        )
    else:
        raise ValueError(
            "Invalid optimizer_type. Expected 'lion', 'adamw', 'deepspeed' or 'stable_adamw', got: {}".format(
                optimizer_type
            )
        )

    # Return the optimizer.
    return optimizer
