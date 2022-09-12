from fastapi import FastAPI
from pydantic import BaseModel
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
	return {"result": result}
