import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.interpretability_attention_analysis import denormalize_image

activations = None
gradients = None

def forward_hook_activations(module, input, output):
    """
    Forward hook to capture the activations (output) from a target layer.

    Args:
        module (nn.Module): The layer module where the hook is registered.
        input (tuple): Input to the layer.
        output (Tensor or tuple): Output from the layer.

    Returns:
        None. Stores the layer output globally in activations.
    """
    global activations
    activations = output[0] if isinstance(output, tuple) else output

def backward_hook_gradients(module, grad_input, grad_output):
    """
    Backward hook to capture gradients flowing through a target layer.

    Args:
        module (nn.Module): The layer module where the hook is registered.
        grad_input (tuple): Gradient inputs to the layer.
        grad_output (tuple or Tensor): Gradient outputs from the layer.

    Returns:
        None. Stores the gradients in gradients globally.
    """
    global gradients
    gradients = grad_output[0] if isinstance(grad_output, tuple) else grad_output

def register_hooks(model, layer):
    """
    Registers forward and backward hooks on a given model layer to track activations and gradients.

    Args:
        model (nn.Module): The model containing the target layer.
        layer (nn.Module): The target layer to hook.

    Returns:
        tuple: Forward hook handle and backward hook handle.
    """
    f_hook = layer.register_forward_hook(forward_hook_activations)
    b_hook = layer.register_full_backward_hook(backward_hook_gradients)
    return f_hook, b_hook

def compute_gradcam(model, input_ids, attention_mask, pixel_values, target_layer):
    """
    Computes Grad-CAM for a multimodal or unimodal input (None), using a target layer in the model.

    Args:
        model (nn.Module): The trained multimodal model.
        input_ids (Tensor): Input token IDs for text modality, shape (1, seq_len).
        attention_mask (Tensor): Attention mask for text modality, shape (1, seq_len).
        pixel_values (Tensor): Input image tensor, shape (1, 3, H, W).
        target_layer (nn.Module): The layer on which hooks are registered to extract activations and gradients.

    Returns:
        Tuple:
            - cam (ndarray): Grad-CAM heatmap upsampled to (224, 224).
            - pred_class (int): Predicted class index for the input.
    """
    global activations, gradients
    activations, gradients = None, None  # Clear previous state

    # Register hooks
    f_hook, b_hook = register_hooks(model, target_layer)

    model.eval()
    pixel_values.requires_grad_(True)

    logits = model(input_ids, attention_mask, pixel_values)
    pred_class = logits.argmax(dim=1).item()
    score = logits[0, pred_class]

    score.backward()

    # Global average pooling of gradients
    weights = gradients.mean(dim=1, keepdim=True)  # shape: (1, 1, D)
    cam = (weights * activations).sum(dim=-1)      # shape: (1, L)

    # ReLU and normalize
    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= cam.max()

    # Reshape and upsample
    L = cam.shape[-1]
    if (L**0.5).is_integer():
        H = W = int(L**0.5)
    else:
        H, W = 10, 5
    cam = cam.reshape(1, 1, H, W)
    cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().detach().numpy()

    # Clean up hooks
    f_hook.remove()
    b_hook.remove()

    return cam, pred_class

def plot_gradcam_grid(model, results, target_layer, name, examples_per_class=3):
    """
    Plots a grid of Grad-CAM image heatmaps for multiple examples across prediction categories.

    Args:
        model (nn.Module): The trained multimodal model.
        results (dict): Dictionary with keys "TP", "TN", "FP", "FN", each mapping to a dict
                        containing tensors for "input_ids", "attention_mask", and "pixel_values".
        target_layer (nn.Module): Layer for Grad-CAM extraction.
        name (str): Filename to save the resulting plot (PNG format).
        examples_per_class (int): Number of examples to visualize per category.

    Returns:
        None. Saves and displays a matplotlib plot.
    """
    model.eval()
    categories = ["TP", "TN", "FP", "FN"]
    num_rows = len(categories)
    num_cols = examples_per_class

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))

    for row_idx, category in enumerate(categories):
        data = results.get(category, {})
        input_ids = data.get("input_ids")
        attention_mask = data.get("attention_mask")
        pixel_values = data.get("pixel_values")

        if input_ids is None or attention_mask is None or pixel_values is None:
            print(f"Skipping {category}: missing data.")
            continue

        num_examples = min(examples_per_class, input_ids.shape[0])

        for i in range(num_examples):
            input_id = input_ids[i].unsqueeze(0)
            attention = attention_mask[i].unsqueeze(0)
            pixel = pixel_values[i].unsqueeze(0)
            
            cam, pred_class = compute_gradcam(model, input_id, attention, pixel, target_layer)

            image = denormalize_image(pixel[0]).permute(1, 2, 0).cpu().detach().numpy()

            ax = axs[row_idx][i] if num_cols > 1 else axs[row_idx]
            ax.imshow(image)
            ax.imshow(cam, cmap='jet', alpha=0.5)
            ax.set_title(f"{category} Example {i+1}")
            ax.axis('off')

    plt.tight_layout()
    plt.suptitle("Grad-CAM Visualization", fontsize=14)
    plt.subplots_adjust(top=0.92)
    plt.savefig(f"interpretability results/plots/{name}.png", dpi=300, bbox_inches='tight')
    plt.show()
