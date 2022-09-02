import torch
from torch import nn

def create_meta(X, models, n_features):
	X_meta = []
	for model in models:
		with torch.no_grad():
			# Linear models get 1D data
			if model.layer1._get_name() == 'Linear':
				X_meta.append(model.get_features(X))
			else:
				X_meta.append(model.get_features(X.reshape(-1,1,n_features)))
      	
	meta = X_meta[0]
	for i in range(1, len(X_meta)):
		X_meta = torch.cat([X_meta, X_meta[i]], dim=1)
  
	# append original dataset	
	X_meta.insert(-1,X)
	return meta
