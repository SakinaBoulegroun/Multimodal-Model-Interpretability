from PIL import Image
import torch
from torch.utils.data import Dataset
import os
from collections import defaultdict

class MultimodalDataset(Dataset):
    """
    A PyTorch Dataset for processing and loading multimodal data samples.

    This dataset supports three input modes: "text", "image", and "both".
    It uses a processor (e.g., CLIPProcessor) to tokenize inputs appropriately.
    
    Args:
        df (pd.DataFrame): Contains 'text', 'label', and 'img' columns.
        processor: A HuggingFace processor that can handle text and/or image inputs.
        mode (str): Type of input data ("text", "image", or "both").
    """
    def __init__(self, df, processor, mode): #mode can be 'text', 'image', 'both'
        super().__init__()
        self.df=df
        self.processor=processor
        self.mode=mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['text']
        label = row['label']
        image_path = os.path.join("data",row["img"])


        if self.mode == 'text':
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77  # CLIP's max context length
            )
        elif self.mode == 'image':
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            )
        elif self.mode == 'both':
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # Remove batch dimension (1st dim = 1) returned by processor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Add label
        inputs['labels'] = torch.tensor(label, dtype=torch.long)

        return inputs



def collect_data_for_analysis_per_category(model, dataloader, device):
    """
    Collects model predictions and groups them by prediction correctness category: TP, TN, FP, FN.

    Args:
        model (nn.Module): Trained model used for inference.
        dataloader (DataLoader): DataLoader providing the evaluation dataset.
        device (str): Device on which the model and data should be placed ("cuda" or "cpu").

    Returns:
        results (dict): A dictionary with keys "TP", "TN", "FP", "FN". Each maps to another dictionary 
                        containing input tensors grouped by prediction category.
    """
    model.eval()
    results = {
        "TP": defaultdict(list),
        "TN": defaultdict(list),
        "FP": defaultdict(list),
        "FN": defaultdict(list)
    }

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            pixel_values = batch.get("pixel_values", None)
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)

            # Forward pass
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )
            preds = torch.argmax(logits, dim=1)

            # Categorize each sample
            for i in range(len(labels)):
                true = labels[i].item()
                pred = preds[i].item()

                if true == 1 and pred == 1:
                    category = "TP"
                elif true == 0 and pred == 0:
                    category = "TN"
                elif true == 0 and pred == 1:
                    category = "FP"
                elif true == 1 and pred == 0:
                    category = "FN"
                else:
                    continue  # Skip if invalid

                results[category]["input_ids"].append(input_ids[i].cpu())
                results[category]["attention_mask"].append(attention_mask[i].cpu())
                if pixel_values is not None:
                    results[category]["pixel_values"].append(pixel_values[i].cpu())

    # Stack lists into tensors
    for category in results:
        for key in results[category]:
            results[category][key] = torch.stack(results[category][key])

    return results

def filter_short_examples_per_category(results, max_tokens=10, max_examples_per_category=3):
    """
    Filters the results dictionary to only include up to `max_examples_per_category`
    per category (TP, TN, FP, FN) with input length <= `max_tokens`.
    
    Args:
        results (dict): Original results dictionary with tensors per category from collect_data_for_analysis_per_category function.
        max_tokens (int): Maximum allowed token length (based on attention mask).
        max_examples_per_category (int): Number of examples to keep per category.

    Returns:
        dict: Filtered results dictionary.
    """
    from collections import defaultdict

    filtered_results = {
        "TP": defaultdict(list),
        "TN": defaultdict(list),
        "FP": defaultdict(list),
        "FN": defaultdict(list)
    }

    for category in results:
        if "attention_mask" not in results[category]:
            continue

        attn_masks = results[category]["attention_mask"]
        total = attn_masks.size(0)

        selected_indices = []
        for i in range(total):
            input_len = (attn_masks[i] == 1).sum().item()
            if input_len <= max_tokens:
                selected_indices.append(i)
            if len(selected_indices) >= max_examples_per_category:
                break

        for key in results[category]:
            filtered_results[category][key] = torch.stack([results[category][key][i] for i in selected_indices])

    return filtered_results

