from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Data(BaseModel):
	columns: list
	data: list

@app.post("/detect")
def detect(X:Data):
	print(X)
	return {"result": "Hello World"}
