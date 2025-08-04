import torch
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os

from models.classifiers import BinaryClassifier  
from src.config import BimodalConfig
from src.dataloader import MultimodalDataset, collect_data_for_analysis_per_category
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from src.interpretability_grad_cam import plot_gradcam_grid


# SetUp

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
bimodal_config=BimodalConfig()
model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer=processor.tokenizer

test_df=pd.read_json("data/original_test.jsonl", lines=True)

test_dataset=MultimodalDataset(test_df, processor, bimodal_config.mode)
test_loader=DataLoader(test_dataset, batch_size=bimodal_config.batch_size,shuffle=True,num_workers=0)

bimodal_model=BinaryClassifier(bimodal_config)
bimodal_model.load_state_dict(torch.load("models/checkpoints/bimodal_classifier.pth"))

data=collect_data_for_analysis_per_category(bimodal_model, test_loader, bimodal_config.device )

target_layer = bimodal_model.model.vision_model.encoder.layers[-1].mlp.fc2

# Plot Grad-CAM heatmaps overlayed on images for examples across prediction categories
plot_gradcam_grid(
    model=bimodal_model,
    results=data,
    target_layer=target_layer,
    name="gradcam_grid_image"
)