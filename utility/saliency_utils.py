import os
import matplotlib.pyplot as plt
import torch
import numpy as np


def get_module_by_name(model, module_name):
    """
    Get a module by its name from a model.
    """
    modules = module_name.split(".")
    mod = model
    for module in modules:
        mod = getattr(mod, module)
    return mod


def plot_saliency_map(
    model,
    input_tensor,
    target_class=None,
    epoch=None,
    batch_num=None,
    model_name="model",
    dataset_name="dataset_0",
):
    """
    Generates a saliency map for the given input tensor and model.

    Args:
        model (torch.nn.Module): The trained model.
        input_tensor (torch.Tensor): The input tensor for which the saliency map is to be generated.
        target_class (int, optional): The target class index. If None, uses the predicted class.
        epoch (int, optional): The epoch number for the filename.
        batch_num (int, optional): The batch number for the filename.
        model_name (str, optional): The name of the model part ('model_a' or 'model_b').

    Returns:
        None
    """
    model.eval()
    input_tensor.requires_grad_()

    # Forward pass
    output = model(input_tensor)

    # If target_class is not specified, use the predicted class
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # Zero gradients
    model.zero_grad()

    # Backward pass
    target = output[0, target_class]
    target.backward()

    # Get gradients
    saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()

    if saliency.ndim == 3:
        saliency = saliency.mean(
            axis=0
        )  # use max, testing with mean for now to see if the map will change
    # Create directory if it doesn't exist
    directory = f"saliency_maps/{dataset_name}/epoch{epoch}/batch{batch_num}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Plot and save the saliency map
    plt.figure(figsize=(10, 10))
    plt.imshow(saliency, cmap="hot")
    plt.title(f"Saliency Map - {model_name}")
    plt.axis("off")
    filename = os.path.join(directory, f"saliency_map_{model_name}.png")
    plt.savefig(filename)
    plt.close()


def plot_weights(
    model,
    layer_name,
    epoch=None,
    batch_num=None,
    model_name="model",
    dataset_name="dataset_0",
):
    """
    Plots the weights of the specified layer and saves them as a PNG file.

    Args:
        model (torch.nn.Module): The trained model.
        layer_name (str): The name of the layer whose weights are to be plotted.
        epoch (int, optional): The epoch number for the filename.
        batch_num (int, optional): The batch number for the filename.
        model_name (str, optional): The name of the model part ('model_a' or 'model_b').

    Returns:
        None
    """
    layer = get_module_by_name(model, layer_name)
    weights = layer.weight.data.cpu().numpy()

    # Create directory if it doesn't exist
    directory = f"weights_plots/{dataset_name}/epoch{epoch}/batch{batch_num}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Determine the number of filters
    num_filters = weights.shape[0]

    # Determine the grid size for subplots
    grid_size = int(np.ceil(np.sqrt(num_filters)))

    # Plot and save the weights
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()

    for i in range(num_filters):
        if len(weights.shape) == 4:  # Conv2d layer
            filter_weights = weights[i, 0, :, :]
        else:  # Linear layer
            filter_weights = weights[i, :]

        axes[i].imshow(filter_weights, cmap="viridis")
        axes[i].axis("off")

    plt.suptitle(f"Weights - {model_name} - Layer: {layer_name}")
    filename = os.path.join(
        directory, f"{model_name}_layer_{layer_name.replace('.', '_')}_weights.png"
    )
    plt.savefig(filename)
    plt.close()


def plot_gradcam(
    model,
    input_tensor,
    target_layer_name,
    target_class=None,
    epoch=None,
    batch_num=None,
):
    """
    Generates a Grad-CAM visualization for the given input tensor and model.

    Args:
        model (torch.nn.Module): The trained model.
        input_tensor (torch.Tensor): The input tensor for which the Grad-CAM is to be generated.
        target_layer_name (str): The target layer name.
        target_class (int, optional): The target class index. If None, uses the predicted class.
        epoch (int, optional): The epoch number for the filename.
        batch_num (int, optional): The batch number for the filename.

    Returns:
        Grad-CAM visualization as a numpy array.
    """

    def hook_fn(module, input, output):
        model.activations = output
        output.register_hook(lambda grad: setattr(model, "activations_grad", grad))

    # Register hook to the target layer
    target_layer = get_module_by_name(model, target_layer_name)
    hook = target_layer.register_forward_hook(hook_fn)

    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad_()

    # Forward pass
    output = model(input_tensor)

    # If target_class is not specified, use the predicted class
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # Zero gradients
    model.zero_grad()

    # Backward pass
    target = output[0, target_class]
    target.backward()

    # Get gradients and activations
    gradients = model.activations_grad
    activations = model.activations

    # Ensure gradients and activations are on the same device
    gradients = gradients.to(device)
    activations = activations.to(device)

    # Check dimensions of gradients and activations
    if gradients.dim() == 2:
        gradients = gradients.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    if activations.dim() == 2:
        activations = activations.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    # Calculate weights
    weights = torch.mean(gradients, dim=[0, 2, 3])

    # Calculate Grad-CAM
    grad_cam = torch.zeros(activations.shape[2:], device=device)
    for i, w in enumerate(weights):
        grad_cam += w * activations[0, i, :, :]

    grad_cam = torch.relu(grad_cam)
    epsilon = 1e-8  # Small constant to avoid division by zero
    grad_cam = grad_cam / torch.max(grad_cam + epsilon)

    # Remove hook
    hook.remove()

    return grad_cam.detach().cpu().numpy()


def plot_gradcams_for_layers(
    model,
    input_tensor,
    layers,
    epoch=None,
    batch_num=None,
    model_name="model",
    dataset_name="dataset_0",
):
    """
    Generates Grad-CAM visualizations for each layer and saves them as separate PNG files in a directory.

    Args:
        model (torch.nn.Module): The trained model.
        input_tensor (torch.Tensor): The input tensor for which the Grad-CAM is to be generated.
        layers (list of str): List of layer names for which to generate Grad-CAM.
        epoch (int, optional): The epoch number for the filename.
        batch_num (int, optional): The batch number for the filename.
        model_name (str, optional): The name of the model part ('model_a' or 'model_b').

    Returns:
        None
    """
    # Create directory if it doesn't exist
    directory = f"gradcam_plots/{dataset_name}/epoch{epoch}/batch{batch_num}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, layer in enumerate(layers):
        grad_cam = plot_gradcam(
            model,
            input_tensor,
            target_layer_name=layer,
            epoch=epoch,
            batch_num=batch_num,
        )
        plt.figure(figsize=(10, 10))
        plt.imshow(grad_cam, cmap="jet")
        plt.title(f"Grad-CAM - {model_name} - Layer: {layer}")
        plt.axis("off")
        filename = os.path.join(
            directory, f"{model_name}_layer_{layer.replace('.', '_')}.png"
        )
        plt.savefig(filename)
        plt.close()

        # Plot and save the weights of the layer
        plot_weights(
            model,
            layer,
            epoch=epoch,
            batch_num=batch_num,
            model_name=model_name,
            dataset_name=dataset_name,
        )
        plot_saliency_map(
            model,
            input_tensor,
            epoch=epoch,
            batch_num=batch_num,
            model_name=model_name,
            dataset_name=dataset_name,
        )
