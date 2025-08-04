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
   git clone <repository_url>
   cd <repository_folder>
   
2. **Install dependencies**
Install required Python packages from requirements.txt:
```bash
pip install -r requirements.txt```

3. **Install dependencies**
**Install the package in editable mode**
This allows you to import and use the code as modules:

bash
Copier
Modifier
pip install -e .



