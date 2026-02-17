import torch 
import torch.nn as nn
from transformers import DistilBertModel

class EmotionClassifier(nn.Module):
  def __init__(self,n_features):
    super(EmotionClassifier,self).__init__()
    self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
    self.dropout = nn.Dropout(0.3)
    self.Linear = nn.Linear(self.bert.config.hidden_size,n_features)

  def forward(self,input_id,attention_mask):
    outputs = self.bert(
        input_ids = input_id,
        attention_mask = attention_mask
    )

    pooled_layer = outputs.last_hidden_state[:,0,:]
    dropout_layer = self.dropout(pooled_layer)
    logits = self.Linear(dropout_layer)

    return logits