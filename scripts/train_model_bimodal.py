from src.train import train_model
from src.config import BimodalConfig
from src.utils import save_model
import pandas as pd
from models.classifiers import BinaryClassifier
from transformers import CLIPProcessor


# SetUp for bimodal model training 

training_df=pd.read_csv("data/training_sample_bigger.csv")

config=BimodalConfig()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
classifier_model=BinaryClassifier(config)

# Train the classifier model using the training dataframe and processor
train_model(classifier_model, training_df, processor, config)

# Save the trained model to a file
save_model(classifier_model, "bimodal_classifier.pth")

print("The model has been effectively trained and saved")