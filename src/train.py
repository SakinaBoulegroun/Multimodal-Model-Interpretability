import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from src.dataloader import MultimodalDataset

def train_model(model, df, processor, config):
    """
    Trains a BinaryClassifier model with accuracy and using data from a DataFrame.

    Args:
        model (nn.Module): The PyTorch model to train.
        df (pd.DataFrame): DataFrame containing the training data.
        processor: A preprocessing utility (e.g., tokenizer, feature extractor).
        config (Namespace): Configuration object with training parameters:
            - lr (float): Learning rate.
            - batch_size (int): Size of training batches.
            - epochs (int): Number of training epochs.
            - mode (str): "text", "image", or "bimodal".

    Returns:
        None
    """
    criterion=nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), config.lr)
    
    train_dataset=MultimodalDataset(df, processor,config.mode)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0)
    
    device = next(model.parameters()).device
    
    model.train()

    for epoch in range(config.epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            # Move batch to the same device as the model
            batch = {k: v.to(device) for k, v in batch.items()}

            input_ids = batch.get("input_ids", None)
            attention_mask = batch.get("attention_mask", None)
            pixel_values = batch.get("pixel_values", None)
            labels = batch["labels"]

            # Zero the gradients
            optimizer.zero_grad()

            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           pixel_values=pixel_values)

            # Compute loss
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        accuracy = correct / total

        print(f"Epoch {epoch+1}/{config.epochs} - Loss: {epoch_loss:.4f} - Accuracy: {accuracy:.4f}")
    

    
