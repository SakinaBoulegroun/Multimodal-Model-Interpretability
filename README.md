# Multimodal Model Interpretability

This project focuses on training and interpreting multimodal classifiers that combine image and text data. It includes training scripts, saved model checkpoints, evaluation, and interpretability tools.

## Project Structure

- **data/**  
  Contains the required bimodal data of hateful memes.  
  **Note:** The image files are **not included** in this repository.  
  Please follow the download instructions provided later in this README to download the images from the designated drive.  
  After downloading, place the `img` folder inside the `data` directory.

- **models/**  
  - `classifiers.py`: Defines the model classes for our classifiers.  
  - `checkpoints/`: Folder where trained model checkpoints will be saved during training.

- **src/**  
  Contains various Python scripts defining functions used throughout the project, including:  
  - Data processing  
  - Model training  
  - Configuration settings  
  - Evaluation  
  - Interpretability analysis

- **scripts/**  
  Execution scripts for:  
  - Generating sample data (already registered in the `data` folder)  
  - Training models  
  - Obtaining evaluation metrics

- **interpretability_results/**  
  Stores results from interpretability analyses such as attention weights, Grad-CAM, and logit lens methods.  
  - `plots/`: Contains saved visualization plots from attention weights and Grad-CAM analyses.

- **setup.py**  
  Useful for installing the package in editable mode.

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/SakinaBoulegroun/Multimodal-Model-Interpretability.git
   cd Multimodal-Model-Interpretability
   
2. **Install dependencies**
Install required Python packages from requirements.txt:
```bash
pip install -r requirements.txt
```

3. **Install the package in editable mode**
This allows you to import and use the code as modules:
```bash
pip install -e .
```
4. **Download and prepare image data**
The bimodal hateful memes dataset images are not included in this repository due to size constraints.
Please follow the instructions below to download the images from the provided drive:
- Download the img folder from the following link drive: https://drive.google.com/drive/folders/1-7rruhu1HUogTxS5DTYxRO4Pt1fkbLZy?usp=sharing
- Place the downloaded img folder inside the data/ directory so that the path is:
```bash
data/img/
```

## Usage

**Data**
The `sample_data.csv` file has already been generated, is available in `data` folder and contains 500 positive and 500 negative training points due to computational limits.  

If you want to train your model with more data, modify the scripts so that when loading the data, you do **not** use `training_sample.csv`. Instead, load the full dataset directly from the original JSONL file by using:  
```python
df = pd.read_json(data/"original_train.jsonl", lines=True)
```

**Train the models**
Run the training script to train the classifiers and save the checkpoints in models/checkpoints:
```bash
python scripts/train_model_bimodal.py
python scripts/train_model_text.py
python scripts/train_image.py
```

**Evaluation**
Run the evaluation script to get the metrics. Change the config and the name of the model depending on which modality you are focusing on.
```bash
python scripts/evaluation.py
```

**Interpretability Analysis**

- **Attention Weights**  
  Two main functions are used to analyze the attention matrices:  
  1. One function computes the attention matrix averaged over all heads in the last attention layer, separately for image and text modalities. This is done for true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN) to compare attention patterns across these different categories.  
  2. Another function shows the evolution of the attention matrix across layers or time, providing insight into how attention changes as information flows through the model.  

  For both attention weights and Grad-CAM, the code includes functionality to plot and save the resulting figures.

- **Grad-CAM**  
  Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique to visualize which regions in an image contribute most to the model's decision by using gradients flowing into the last convolutional layers.  
  In this project, Grad-CAM generates a grid of heatmaps overlayed on the input images based on the last attention layer. You can change the layer used for Grad-CAM visualization by modifying the `target_layer` parameter.  

- **Logit Lens**  
  The logit lens analysis tracks the predicted logits for a chosen token (default is the [CLS] token) across different layers of the model. This shows how the model’s prediction evolves layer by layer, helping to interpret the model’s internal decision-making process.

All generated plots from these analyses are saved in the `interpretability_results/plots/` folder for easy access and review.




