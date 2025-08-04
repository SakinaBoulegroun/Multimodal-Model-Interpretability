from src.dataloader import MultimodalDataset
from models.classifiers import BinaryClassifier
from src.config import BimodalConfig, ImageConfig
from src.evaluate import evaluate_model
import pandas as pd
from transformers import CLIPProcessor
import torch 
from torch.utils.data import DataLoader

# SetUp for Model evaluation

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

df_bimodal_test=pd.read_json("data/original_test.jsonl", lines=True)
bimodal_config=ImageConfig()

bimodal_classifier_model=BinaryClassifier(bimodal_config)
bimodal_classifier_model.load_state_dict(torch.load("models/checkpoints/image_classifier.pth"))

test_dataset=MultimodalDataset(df_bimodal_test, processor, bimodal_config.mode)
test_dataloader= DataLoader(
    test_dataset,
    batch_size=bimodal_config.batch_size,
    shuffle=True,
    num_workers=0
)

# Evaluate the selected classifier on the test data and print results
print(evaluate_model(bimodal_classifier_model, test_dataloader, bimodal_config.device))

