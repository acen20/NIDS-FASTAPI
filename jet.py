## Framework/Tools
import torch
from torch import nn

## Modules
from wings import CNN, LSTM
from tail import MLP
from nose import MLP as nMLP
from body import create_meta
from utils import preprocess_data

## Utilities
import pickle


def load_objects():
	## Load objects
	### loading standard scaler object
	with open('objs/scaler.pkl','rb') as f:
		ss = pickle.load(f)
		
	### loading label encoder object	
	with open('objs/le.pkl','rb') as f:
		enc = pickle.load(f)
	return ss, enc

def load_wings_weights(NUM_FEATURES, NUM_CLASSES, device):
	## Load wings weights
	lstm = LSTM(input_size=NUM_FEATURES,hidden_size=128,num_layers=2,
			  num_classes=NUM_CLASSES, dropout=0, device=device).to(device)
	lstm.load_state_dict(torch.load('models/LSTM.pt'))

	cnn = CNN(in_channels=1,out_channels=3,hidden_size=64,
			input_size=NUM_FEATURES,num_classes=NUM_CLASSES,kernel_size=5,
			dropout=0.2).to(device)
	cnn.load_state_dict(torch.load('models/CNN.pt'))
	return lstm, cnn


def load_tail_weights(NUM_FEATURES, NUM_CLASSES, device):
	## Load tail weights
	mlp = MLP(input_size=NUM_FEATURES, hidden_size=80, num_classes=NUM_CLASSES, 
			dropout=0).to(device)
	mlp.load_state_dict(torch.load('models/MLP.pt'))
	return mlp
	
def load_nose_weights(NUM_FEATURES, NUM_CLASSES, device):
	## Load MLP weights
	mlp = nMLP(520, 128, NUM_CLASSES, 0)
	mlp.load_state_dict(torch.load('models/DEEP_ENSEMBLE.pt'))
	return mlp
	

def set_eval_mode(lstm, cnn, mlp, nmlp):
	## Set models to evaluation state
	lstm.eval()
	cnn.eval()
	mlp.eval()
	nmlp.eval()

def get_predictor(device):
	NUM_FEATURES = 78
	NUM_CLASSES = 7
	
	# loading wings weights
	lstm, cnn = load_wings_weights(NUM_FEATURES, NUM_CLASSES, device)
	mlp = load_tail_weights(NUM_FEATURES, NUM_CLASSES, device)
	nmlp = load_nose_weights(NUM_FEATURES, NUM_CLASSES, device)
	
	set_eval_mode(lstm, cnn, mlp, nmlp)
	
	models = [lstm, cnn, mlp]
	clf = nmlp
	
	# loading standardscaler and labelecnoder objects
	ss, enc = load_objects()
	
	objects = {
		'scaler':ss,
		'labelencoder':enc
	}
	
	return models, clf, objects, NUM_FEATURES
	

def predict(data, models, clf, objects, n_features):
	## rearrange input features and scale data
	X = preprocess_data(data.data, data.columns, objects)
	
	## Convert to tensor
	X = torch.tensor(X, dtype=torch.float)
	
	## Create Meta
	X = create_meta(X, models, n_features=n_features)
	
	## Infer
	pred = clf(X).cpu().detach()
	pred = torch.argmax(pred).numpy()
	pred = objects['labelencoder'].inverse_transform([pred])
	print(pred)
	return pred
