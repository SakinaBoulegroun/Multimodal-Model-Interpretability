import torch
from models.classifiers import BinaryClassifier
import os

def load_model(checkpoint_path, config, device=None):
    """
    Loads a trained BinaryClassifier from a checkpoint.
    
    Args:
        checkpoint_path (str): Path to the .pth file
        modality (str): "text", "image", or "bimodal"
        device (str): "cuda" or "cpu" (optional)

    Returns:
        model (nn.Module): Loaded model ready for inference
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = BinaryClassifier(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model


def save_model(model, name):
    """
    Saves the state dictionary of a trained model to a checkpoint file.

    Args:
        model (nn.Module): The trained PyTorch model to save.
        name (str): The name of the checkpoint file (e.g., "model.pth").

    Returns:
        None
    """
    path = os.path.join("models", "checkpoints")
    os.makedirs(path, exist_ok=True)  # Ensure directory exists
    torch.save(model.state_dict(), os.path.join(path, name))


