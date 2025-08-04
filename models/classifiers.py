import torch.nn as nn
import torch
from transformers import CLIPModel

class BinaryClassifier(nn.Module):
  """
  Binary classifier built on top of a frozen pre-trained CLIP model.

  Depending on the mode, it uses text embeddings, image embeddings, or their concatenation
  as input features for a simple feedforward classifier head.

  Args:
      config: Configuration object with the following attributes:
          - embedding_dimension (int): Dimensionality of the CLIP embeddings (text/image).
          - hidden_dimension (int): Number of hidden units in the classifier MLP.
          - mode (str): One of {"text", "image", "both"} indicating which embeddings to use.

  Attributes:
      model (CLIPModel): Pre-trained CLIP model with frozen weights.
      classifier (nn.Sequential): Feedforward neural network classifier head.

  Forward Args:
      input_ids (torch.LongTensor, optional): Token ids for text input, shape (batch_size, seq_len).
      attention_mask (torch.LongTensor, optional): Attention mask for text input, same shape as input_ids.
      pixel_values (torch.FloatTensor, optional): Image tensor input, shape (batch_size, 3, H, W).

  Returns:
      torch.FloatTensor: Logits tensor of shape (batch_size, 2) for binary classification.
  """
  def __init__(self, config):
    super().__init__()
    self.model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    self.embedding_dim=config.embedding_dimension
    self.hidden_dim=config.hidden_dimension
    self.mode=config.mode

    for param in self.model.parameters():
            param.requires_grad = False

    if self.mode=="both":
      self.classifier=nn.Sequential(
        nn.Linear(2*self.embedding_dim,self.hidden_dim),
        nn.ReLU(),
        nn.Linear(self.hidden_dim,2)
        )

    if self.mode=="text" or self.mode=="image":
      self.classifier=nn.Sequential(
        nn.Linear(self.embedding_dim,self.hidden_dim),
        nn.ReLU(),
        nn.Linear(self.hidden_dim,2)
        )  

  def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
    embeddings = []

    if input_ids is not None:
        text_emb = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        embeddings.append(text_emb)

    if pixel_values is not None:
        image_emb = self.model.get_image_features(pixel_values=pixel_values)
        embeddings.append(image_emb)

    if len(embeddings) > 1:
        combined = torch.cat(embeddings, dim=1)
    else:
        combined = embeddings[0]

    logits = self.classifier(combined)
    return logits
      
    

