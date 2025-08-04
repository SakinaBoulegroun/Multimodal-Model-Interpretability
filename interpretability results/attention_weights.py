import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from src.config import BimodalConfig
from models.classifiers import BinaryClassifier
from src.dataloader import collect_data_for_analysis_per_category, MultimodalDataset, filter_short_examples_per_category
from src.interpretability_attention_analysis import plot_average_attention_grid, plot_evolution_attention_heads

#SetUp

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
bimodal_config=BimodalConfig()
model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer=processor.tokenizer

test_df=pd.read_json("data/original_test.jsonl", lines=True)

test_dataset=MultimodalDataset(test_df, processor, bimodal_config.mode)
test_loader=DataLoader(test_dataset, batch_size=bimodal_config.batch_size,shuffle=True,num_workers=0)

bimodal_model=BinaryClassifier(bimodal_config)
bimodal_model.load_state_dict(torch.load("models/checkpoints/bimodal_classifier.pth"))

#For average attention weights: 4 by 3 grid
data_long=collect_data_for_analysis_per_category(bimodal_model, test_loader, bimodal_config.device)
data=filter_short_examples_per_category(data_long, max_tokens=15, max_examples_per_category=3)

#For evolution attention weights: unique datapoint for visualization
first_batch=next(iter(test_loader))
datapoint = {
    "input_ids": first_batch["input_ids"][0],           # shape: (seq_len,)
    "attention_mask": first_batch["attention_mask"][0], # shape: (seq_len,)
    "pixel_values": first_batch["pixel_values"][0],     # shape: (C, H, W)
    "labels": first_batch["labels"][0]                  # scalar
}


# Plot Average Attention Weigths: Text examples
#plot_average_attention_grid(model, tokenizer, data, "text", "average_attention_grid_text", examples_per_class=3)

# Plot Average Attention Weights: Image examples
#plot_average_attention_grid(model, tokenizer, data, "image", "average_attention_grid_image", examples_per_class=3)

# Plot Evolution Attention Weights: Text example
plot_evolution_attention_heads(model, tokenizer,datapoint, "evolution_attention_grid_text", "text")

# Plot Evolution Attention Weights: Text example
#plot_evolution_attention_heads(model, tokenizer,datapoint, "evolution_attention_grid_image", "image")