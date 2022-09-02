## Framework/Tools
import torch
from torch import nn

## Modules
from wings import CNN, LSTM
from tail import MLP
from nose import MLP as nMLP
from body import create_meta

## Utilities
import pickle

NUM_FEATURES = 78
NUM_CLASSES = 7

## Load objects
### loading standard scaler object
with open('objs/scaler.pkl','rb') as f:
	ss = pickle.load(f)
	
### loading label encoder object	
with open('objs/le.pkl','rb') as f:
	enc = pickle.load(f)

## Load wings weights
lstm = LSTM(input_size=NUM_FEATURES,hidden_size=128,num_layers=2,
	          num_classes=NUM_CLASSES, dropout=0).to(device)
lstm.load_state_dict(torch.load('models/LSTM.pt'))

cnn = CNN(in_channels=1,out_channels=3,hidden_size=64,
	        input_size=NUM_FEATURES,num_classes=NUM_CLASSES,kernel_size=5,
	        dropout=0.1).to(device)
cnn.load_state_dict(torch.load('models/CNN.pt'))


## Load tail weights
mlp = MLP(input_size=NUM_FEATURES, hidden_size=80, num_classes=NUM_CLASSES, 
	        dropout=0).to(device)
mlp.load_state_dict(torch.load('models/MLP.pt'))

## Set models to evaluation state
lstm.eval()
cnn.eval()
mlp.eval()

## Create Meta
X = create_meta(X, [lstm, cnn, mlp], n_features=NUM_FEATURES)

## Load MLP weights
mlp = nMLP(520, 128, NUM_CLASSES, 0)
mlp.load_state_dict(torch.load('models/DEEP_ENSEMBLE.pt'))

## Set classifier to evaluation state
mlp.eval()

def predict(X):
	## Extract features
	## Convert to tensor
	X = torch.tensor(X)
	## Infer
	pred = mlp(X).cpu().detach()
	pred = enc.inverse_transform()
