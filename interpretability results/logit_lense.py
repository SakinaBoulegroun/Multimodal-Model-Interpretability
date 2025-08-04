import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from src.config import TextConfig
from models.classifiers import BinaryClassifier
from src.dataloader import MultimodalDataset
from src.interpretability_logit_lense import get_logit_lens_outputs

# SetUp

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_config=TextConfig()
model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer=processor.tokenizer

test_df=pd.read_json("data/original_test.jsonl", lines=True)

test_dataset=MultimodalDataset(test_df, processor, text_config.mode)
test_loader=DataLoader(test_dataset, batch_size=text_config.batch_size,shuffle=True,num_workers=0)

text_model=BinaryClassifier(text_config)
text_model.load_state_dict(torch.load("models/checkpoints/text_classifier.pth"))

first_batch=next(iter(test_loader))
datapoint = {
    "input_ids": first_batch["input_ids"][0],           # shape: (seq_len,)
    "attention_mask": first_batch["attention_mask"][0], # shape: (seq_len,)     
    "labels": first_batch["labels"][0]                  # scalar
}

# Get the logits from each transformer encoder layer's CLS token representation
print(get_logit_lens_outputs(text_model, text_model.classifier, datapoint, text_model.model.text_model.encoder.layers, text_config.device))
