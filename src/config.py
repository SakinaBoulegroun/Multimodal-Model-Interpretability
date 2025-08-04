from transformers import CLIPProcessor
import torch

class BimodalConfig:
    """
    Configuration for training a model using both text and image inputs.
    """
    mode="both"
    embedding_dimension=512
    hidden_dimension=500
    batch_size=32
    epochs=10
    lr=1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextConfig:
    """
    Configuration for training a model using text-only inputs.
    """
    mode="text"
    embedding_dimension=512
    hidden_dimension=500
    batch_size=32
    epochs=10
    lr=1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageConfig:
    """
    Configuration for training a model using image-only inputs.
    """
    mode="image"
    embedding_dimension=512
    hidden_dimension=500
    batch_size=32
    epochs=10
    lr=1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

