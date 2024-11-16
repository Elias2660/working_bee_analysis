import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def get_layer_by_name(model, layer_name):
    """
    Retrieves a layer from the model based on a dot-separated path.
    
    Args:
        model (torch.nn.Module): The model from which to retrieve the layer.
        layer_name (str): Dot-separated path of the layer (e.g., "model_a.4.0").
    
    Returns:
        torch.nn.Module: The layer corresponding to the provided path.
    """
    if isinstance(layer_name, list):
        layer_name = layer_name[0]  # Use the first item if it's a list

    parts = layer_name.split('.')  # Split the layer name by dots
    layer = model
    for part in parts:
        layer = getattr(layer, part)  # Get the layer attribute
    return layer


def plot_gradcam_for_multichannel_input(model, input_tensor, target_layer_name, model_name, target_classes=None, batch_num=None):
    """
    Generates and saves Grad-CAM overlays for each channel in a multi-channel input tensor for the entire batch.

    Args:
        model (torch.nn.Module): The trained model.
        input_tensor (torch.Tensor): The input tensor with shape [batch_size, C, H, W].
        target_layer_name (str): Dot-separated path to the target layer for Grad-CAM.
        model_name (str): Identifier for the model, e.g., "model_a" or "model_b".
        target_classes (list, optional): A list of target class indices for each image in the batch.
        batch_num (int, optional): The batch number for the filename.

    Returns:
        None
    """
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Retrieve the target layer using get_layer_by_name
    target_layer = get_layer_by_name(model, target_layer_name)

    # If target_classes is not provided, use the predicted class for each image
    if target_classes is None:
        with torch.no_grad():
            outputs = model(input_tensor)  # Forward pass for the entire batch
            target_classes = outputs.argmax(dim=1).tolist()  # Get target classes for all images

    # Create the Grad-CAM objects and process each sample in the batch
    targets = [ClassifierOutputTarget(target_class) for target_class in target_classes]
    
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # Convert input tensor to numpy array for overlaying
    input_image = input_tensor.detach().cpu().numpy()
    
    # Process each image in the batch
    for batch_idx in range(input_image.shape[0]):
        # Get the target class for the current image
        target_class = target_classes[batch_idx]

        # Define directory path for each batch, model, and class
        class_directory = f"gradcam_plots/batch_{batch_num}/{model_name}/class_{target_class}"
        os.makedirs(class_directory, exist_ok=True)

        # Process each channel in the image
        for channel_idx in range(input_image.shape[1]):
            channel_image = input_image[batch_idx, channel_idx]  # Shape: (H, W)
            channel_image = (channel_image - channel_image.min()) / (channel_image.max() - channel_image.min())  # Normalize

            # Convert to RGB by repeating the grayscale image across three channels
            channel_image_rgb = np.stack([channel_image] * 3, axis=-1)

            # Get the CAM overlay for the current image
            cam_image = show_cam_on_image(channel_image_rgb, grayscale_cam[batch_idx], use_rgb=True)

            # Plot both the original channel and the Grad-CAM overlay side-by-side
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            
            # Plot the original grayscale channel image
            axs[0].imshow(channel_image, cmap='gray')
            axs[0].set_title(f'Original Channel {channel_idx + 1}')
            axs[0].axis('off')
            
            # Plot the Grad-CAM overlay
            axs[1].imshow(cam_image)
            axs[1].set_title(f'Grad-CAM Overlay for Channel {channel_idx + 1}')
            axs[1].axis('off')
            
            # Save the figure
            filename = os.path.join(class_directory, f"gradcam_overlay_class{target_class}_batch{batch_num}_image{batch_idx}_channel{channel_idx}.png")
            plt.savefig(filename)
            plt.close(fig)
            print(f"Grad-CAM image for image {batch_idx} channel {channel_idx} saved to {filename}")
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def get_layer_by_name(model, layer_name):
    """
    Retrieves a layer from the model based on a dot-separated path.
    
    Args:
        model (torch.nn.Module): The model from which to retrieve the layer.
        layer_name (str): Dot-separated path of the layer (e.g., "model_a.4.0").
    
    Returns:
        torch.nn.Module: The layer corresponding to the provided path.
    """
    if isinstance(layer_name, list):
        layer_name = layer_name[0]  # Use the first item if it's a list

    parts = layer_name.split('.')  # Split the layer name by dots
    layer = model
    for part in parts:
        layer = getattr(layer, part)  # Get the layer attribute
    return layer


def plot_gradcam_for_multichannel_input(model, input_tensor, target_layer_name, model_name, target_classes=None, batch_num=None):
    """
    Generates and saves Grad-CAM overlays for each channel in a multi-channel input tensor for the entire batch.

    Args:
        model (torch.nn.Module): The trained model.
        input_tensor (torch.Tensor): The input tensor with shape [batch_size, C, H, W].
        target_layer_name (str): Dot-separated path to the target layer for Grad-CAM.
        model_name (str): Identifier for the model, e.g., "model_a" or "model_b".
        target_classes (list, optional): A list of target class indices for each image in the batch.
        batch_num (int, optional): The batch number for the filename.

    Returns:
        None
    """
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Retrieve the target layer using get_layer_by_name
    target_layer = get_layer_by_name(model, target_layer_name)

    # If target_classes is not provided, use the predicted class for each image
    if target_classes is None:
        with torch.no_grad():
            outputs = model(input_tensor)  # Forward pass for the entire batch
            target_classes = outputs.argmax(dim=1).tolist()  # Get target classes for all images

    # Create the Grad-CAM objects and process each sample in the batch
    targets = [ClassifierOutputTarget(target_class) for target_class in target_classes]
    
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # Convert input tensor to numpy array for overlaying
    input_image = input_tensor.detach().cpu().numpy()
    
    # Process each image in the batch
    for batch_idx in range(input_image.shape[0]):
        # Get the target class for the current image
        target_class = target_classes[batch_idx]

        # Define directory path for each batch, model, and class
        class_directory = f"gradcam_plots/batch_{batch_num}/{model_name}/class_{target_class}"
        os.makedirs(class_directory, exist_ok=True)

        # Process each channel in the image
        for channel_idx in range(input_image.shape[1]):
            channel_image = input_image[batch_idx, channel_idx]  # Shape: (H, W)
            channel_image = (channel_image - channel_image.min()) / (channel_image.max() - channel_image.min())  # Normalize

            # Convert to RGB by repeating the grayscale image across three channels
            channel_image_rgb = np.stack([channel_image] * 3, axis=-1)

            # Get the CAM overlay for the current image
            cam_image = show_cam_on_image(channel_image_rgb, grayscale_cam[batch_idx], use_rgb=True)

            # Plot both the original channel and the Grad-CAM overlay side-by-side
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            
            # Plot the original grayscale channel image
            axs[0].imshow(channel_image, cmap='gray')
            axs[0].set_title(f'Original Channel {channel_idx + 1}')
            axs[0].axis('off')
            
            # Plot the Grad-CAM overlay
            axs[1].imshow(cam_image)
            axs[1].set_title(f'Grad-CAM Overlay for Channel {channel_idx + 1}')
            axs[1].axis('off')
            
            # Save the figure
            filename = os.path.join(class_directory, f"gradcam_overlay_class{target_class}_batch{batch_num}_image{batch_idx}_channel{channel_idx}.png")
            plt.savefig(filename)
            plt.close(fig)

