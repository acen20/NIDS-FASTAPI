import torch
from torch import nn

class MLP(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes, dropout):
    super(MLP, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_classes = num_classes
    self.dropout = dropout

    self.layer1 = nn.Linear(self.input_size, self.hidden_size)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(self.dropout)
    self.layer2 = nn.Linear(self.hidden_size, self.hidden_size)
    self.fc = nn.Linear(self.hidden_size, self.num_classes)

  def forward(self, x):
    x = self.layer1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.layer2(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc(x)
    x = self.dropout(x)
    return x
  
  def get_features(self, x):
    x = self.layer1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.layer2(x)
    x = self.relu(x)
    return x
