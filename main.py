from fastapi import FastAPI
from pydantic import BaseModel
import warnings
warnings.filterwarnings("ignore")
import torch

# model pipeline
import jet

app = FastAPI()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

models, clf, objects, NUM_FEATURES = jet.get_predictor(device)

class Data(BaseModel):
	columns:list
	data:list[list]

@app.post("/detect")
def detect(data: Data):
	result = jet.predict(data, models, clf, objects, NUM_FEATURES)
	return {
		"result": [0],
		"src_port": 2313,
		"src_ip": "1.421.212.21",
		"dst_port":421,
		"dst_ip":"1.3.51.4",
		"probability": [[1]]	
	}
