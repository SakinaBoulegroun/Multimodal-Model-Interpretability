import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import math
import os
import numpy as np
from torchvision.transforms.functional import normalize, to_pil_image
from torchvision.utils import make_grid
import torchvision.transforms as T
from PIL import Image
import matplotlib.cm as cm

def denormalize_image(tensor):
    """
    Reverses ImageNet normalization on an image tensor for visualization.

    Args:
        tensor (torch.Tensor): Normalized image tensor of shape (3, H, W).

    Returns:
        torch.Tensor: Denormalized image tensor suitable for display.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def plot_average_attention_grid(model, tokenizer, results, modality, name, examples_per_class=3):
    """
    Visualizes the average attention maps from the last attention layer of a model.

    Args:
        model (nn.Module): The trained model with accessible vision/text encoder outputs.
        tokenizer: Tokenizer to convert input_ids to tokens (used for text modality).
        results (dict): Dictionary with keys "TP", "TN", "FP", "FN", each mapping to a dict
                        of input tensors (input_ids, attention_mask, pixel_values).
        modality (str): Either "image" or "text", depending on the model part to visualize.
        name (str): Filename to save the resulting plot (PNG format).
        examples_per_class (int): Number of examples to plot per prediction category.

    Returns:
        None. Saves and displays a matplotlib plot.
    """
    model.eval()

    categories = ["TP", "TN", "FP", "FN"]
    num_rows = len(categories)
    num_cols = examples_per_class

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
    if num_rows == 1:
        axs = [axs]

    for row_idx, category in enumerate(categories):
        data = results.get(category, {})
        input_ids = data.get("input_ids")
        attention_mask = data.get("attention_mask")
        pixel_values = data.get("pixel_values", None)

        if input_ids is None or attention_mask is None:
            print(f"Missing data for {category}, skipping.")
            continue

        batch_size = input_ids.shape[0]
        num_examples = min(batch_size, examples_per_class)

        for i in range(num_examples):
            input_id = input_ids[i].unsqueeze(0)
            attention = attention_mask[i].unsqueeze(0)

            if modality == 'image':
                pixel = pixel_values[i].unsqueeze(0)

                with torch.no_grad():
                    vision_outputs = model.vision_model(
                        pixel_values=pixel,
                        output_attentions=True
                    )
                    vision_attn = vision_outputs.attentions[-1].mean(dim=1)[0]  # (num_patches, num_patches)

                # Average attention over all patches (focus on CLS or overall attention?)
                attn_map = vision_attn.mean(dim=0).detach().cpu().numpy()

                # Get CLS attention if available
                cls_attn = vision_attn[0].detach().cpu().numpy()[1:]  # drop CLS if needed

                # Get patch-wise attention score for visualization
                patch_attn = vision_attn[0, 1:].detach().cpu().numpy()  # CLS token's attention to patches
                num_patches = int(np.sqrt(len(patch_attn)))
                patch_attn = patch_attn.reshape(num_patches, num_patches)

                # Get image and resize attention
                image_tensor = denormalize_image(pixel[0]).clamp(0, 1)
                image_np = to_pil_image(image_tensor)

                patch_attn_resized = T.Resize(image_np.size[::-1])(Image.fromarray((patch_attn / patch_attn.max() * 255).astype(np.uint8)))
                patch_attn_resized = np.array(patch_attn_resized) / 255.0

                # Create overlay
                cmap = cm.get_cmap('viridis')
                attn_color = cmap(patch_attn_resized)[..., :3]
                attn_overlay = (np.array(image_np) / 255.0 * 0.5 + attn_color * 0.5)

                ax = axs[row_idx][i] if num_cols > 1 else axs[row_idx]
                ax.imshow(attn_overlay)
                ax.axis("off")
                ax.set_title(f"{category} Example {i+1}")

            elif modality == 'text':
                with torch.no_grad():
                    text_outputs = model.text_model(
                        input_ids=input_id,
                        attention_mask=attention,
                        output_attentions=True
                    )
                    text_attn = text_outputs.attentions[-1].mean(dim=1)[0]

                tokens = tokenizer.convert_ids_to_tokens(input_id[0])
                filtered_indices = [idx for idx, tok in enumerate(tokens) if tok not in ["<|endoftext|>", "</s>", "[PAD]"]]
                tokens = [tokens[idx] for idx in filtered_indices]
                attn_map = text_attn[filtered_indices][:, filtered_indices].detach().cpu().numpy()

                ax = axs[row_idx][i] if num_cols > 1 else axs[row_idx]
                sns.heatmap(attn_map,
                            xticklabels=tokens,
                            yticklabels=tokens,
                            cmap="viridis",
                            ax=ax,
                            cbar=False)
                ax.set_title(f"{category} Example {i+1}")
                ax.tick_params(axis='x', labelrotation=90, labelsize=8)
                ax.tick_params(axis='y', labelrotation=0, labelsize=8)

    plt.tight_layout(pad=2.0, w_pad=2.5, h_pad=3)
    plt.suptitle("Attention Maps - Last Layer Averaged Over Heads", fontsize=13)
    os.makedirs("interpretability results/plots", exist_ok=True)
    plt.savefig(os.path.join("interpretability results/plots", name), bbox_inches='tight', dpi=300)
    plt.show()


def plot_evolution_attention_heads(model, tokenizer, datapoint, name, modality):
    """
    Visualizes the attention maps of the last attention layer's heads for a single data point.

    Args:
        model (nn.Module): The trained model with accessible vision and text encoders.
        tokenizer: Tokenizer to convert input_ids to tokens (used for text modality).
        datapoint (dict): Dictionary containing input tensors. For text modality, expects keys
                          "input_ids" and "attention_mask". For image modality, expects key
                          "pixel_values".
        name (str): Filename to save the resulting plot (PNG format).
        modality (str): Either "image" or "text", specifying which model part's attention to visualize.

    Returns:
        None. Saves the attention head plots to disk and displays them.
    """
    model.eval()

    with torch.no_grad():
        if modality == "text":
            input_ids = datapoint["input_ids"].unsqueeze(0)
            attention_mask = datapoint["attention_mask"].unsqueeze(0)

            outputs = model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            att = outputs.attentions[-1].squeeze(0)  # shape: (num_heads, seq_len, seq_len)
            tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().tolist())

            # Filter out unwanted tokens
            filtered_indices = [i for i, tok in enumerate(tokens) if tok not in ["<|endoftext|>", "</s>", "[PAD]"]]
            filtered_tokens = [tokens[i] for i in filtered_indices]
            num_heads = att.shape[0]

            # Grid size
            cols = min(4, num_heads)
            rows = math.ceil(num_heads / cols)

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
            axes = axes.flatten()

            for head_i in range(num_heads):
                att_matrix = att[head_i][filtered_indices][:, filtered_indices].cpu().numpy()

                ax = axes[head_i]
                sns.heatmap(att_matrix,
                            xticklabels=filtered_tokens,
                            yticklabels=filtered_tokens,
                            cmap="viridis",
                            cbar=False,
                            ax=ax)

                ax.set_title(f"Head {head_i}", fontsize=8)
                ax.tick_params(labelsize=6, rotation=90, axis='x')
                ax.tick_params(labelsize=6, rotation=0, axis='y')

            # Hide unused subplots
            for ax in axes[num_heads:]:
                ax.axis('off')

        elif modality == "image":
            pixel_values = datapoint["pixel_values"].unsqueeze(0)
            outputs = model.vision_model(
                pixel_values=pixel_values,
                output_attentions=True
            )
            att = outputs.attentions[-1].squeeze(0)  # (num_heads, num_patches, num_patches)
            num_heads = att.shape[0]

            # Get original image
            image = denormalize_image(pixel_values[0]).permute(1, 2, 0).cpu().numpy()

            # Get attention for [CLS] token to patches (usually the first row)
            att_cls = att[:, 0, 1:]  # shape: (num_heads, num_patches)
            num_patches = att_cls.shape[-1]
            grid_size = int(math.sqrt(num_patches))

            # Reshape and interpolate attention maps
            att_maps = att_cls.reshape(num_heads, 1, grid_size, grid_size)  # (num_heads, 1, grid, grid)
            att_maps = F.interpolate(att_maps, size=image.shape[:2], mode="bilinear", align_corners=False)

            cols = min(4, num_heads)
            rows = math.ceil(num_heads / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
            axes = axes.flatten()

            for head_i in range(num_heads):
                ax = axes[head_i]
                ax.imshow(image)
                att_map = att_maps[head_i, 0].cpu().numpy()
                ax.imshow(att_map, cmap='jet', alpha=0.5)
                ax.set_title(f"Head {head_i}", fontsize=8)
                ax.axis('off')

            for ax in axes[num_heads:]:
                ax.axis('off')

    plt.tight_layout()
    plt.suptitle("Last Layer Attention Heads", fontsize=12, y=1.02)
    plt.savefig(os.path.join("interpretability results/plots", name), bbox_inches='tight', dpi=300)
    plt.show()
