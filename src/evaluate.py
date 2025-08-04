import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_model(model, dataloader, device):
    """
    Evaluates a trained model on a validation or test dataset.

    Args:
        model (nn.Module): Trained PyTorch model.
        dataloader (DataLoader): DataLoader object for evaluation data.
        device (str): Device on which to run the evaluation ("cuda" or "cpu").

    Returns:
        None
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch.get("input_ids", None)
            attention_mask = batch.get("attention_mask", None)
            pixel_values = batch.get("pixel_values", None)
            labels = batch["labels"].to(device)

            if input_ids is not None:
                input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)

            #loss
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()

            #predictions
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    avg_loss = total_loss / len(dataloader)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"Average Log Loss: {avg_loss:.4f}")

