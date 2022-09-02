from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Data(BaseModel):
	columns: list
	data: list

@app.post("/detect")
def detect(X:Data):
	# drop extra columns
	# sort columns
	
	print(len(X.columns))
	return {"result": ["Hello World"]}
