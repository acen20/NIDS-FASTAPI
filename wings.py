import torch
from torch import nn

## LSTM

class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout, device):
    super(LSTM, self).__init__()
    self.num_layers = num_layers
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_classes = num_classes
    self.dropout = dropout
    self.device = device
    
    self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, 
                        dropout = self.dropout, batch_first=True)
    self.fc = nn.Linear(self.hidden_size, self.num_classes)

  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
    out, _ = self.lstm(x, (h0,c0))
    out = out[:,-1,:]
    out = self.fc(out)
    return out

  def get_features(self, x):
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
    out, _ = self.lstm(x, (h0,c0))
    out = out[:,-1,:]
    return out
    
## CNN
class CNN(nn.Module):
  def __init__(self, in_channels, out_channels, hidden_size, input_size,
       num_classes, kernel_size, dropout):
    super(CNN, self).__init__()

    self.in_channels = in_channels
    self.input_size = input_size
    self.num_classes = num_classes
    self.kernel_size = kernel_size
    self.num_layers = 2
    self.hidden_size = hidden_size
    self.out_channels = out_channels
    self.dropout = dropout

    self.layer1 = nn.Sequential(
        nn.Conv1d(in_channels = self.in_channels, out_channels = self.out_channels,
                  kernel_size=self.kernel_size, stride = 1, padding='same'),
        nn.ReLU(),
        nn.Dropout(self.dropout)
    )
    self.layer2 = nn.Sequential(
        nn.Conv1d(in_channels = self.out_channels, out_channels = self.out_channels,
                  kernel_size=self.kernel_size, stride = 1, padding='same'),
        nn.ReLU(),
        nn.Dropout(self.dropout)
    )
    self.flatten = nn.Flatten()
    self.fc = nn.Sequential(
        nn.Linear(self.input_size*self.out_channels, self.hidden_size),
        nn.ReLU(),
        nn.Dropout(self.dropout),
        nn.Linear(self.hidden_size, self.num_classes)
    )


  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.flatten(x)
    x = self.fc(x)
    return x

  def get_features(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.flatten(x)
    return x
      
      
