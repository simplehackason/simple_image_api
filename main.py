from fastapi import FastAPI
from pydantic import BaseModel
import base64
from typing import Union
from fastapi.responses import JSONResponse

class Item(BaseModel):
    image: str

class Output(BaseModel):
    image: str
    text: str

app = FastAPI()


@app.post("/")
def input(item: Item) -> Output:
    return 
    
    