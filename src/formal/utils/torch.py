from torch import nn


def print_parameter_summary(model: nn.Module) -> None:
    num_parameters = 0
    num_trainable_parameters = 0
    total_memory_bytes = 0

    for parameter in model.parameters():
        num_elements = parameter.numel()
        num_parameters += num_elements
        total_memory_bytes += num_elements * parameter.element_size()

        if parameter.requires_grad:
            num_trainable_parameters += num_elements

    memory_footprint = total_memory_bytes / (1024**2)
    trainable_percentage = 100 * num_trainable_parameters / num_parameters if num_parameters > 0 else 0

    print(
        f"all parameters: {num_parameters:,} || "
        f"trainable parameters: {num_trainable_parameters:,} || "
        f"trainable percentage: {trainable_percentage:.2f} || "
        f"memory footprint: {memory_footprint:.2f} MB"
    )
