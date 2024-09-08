from fastapi import FastAPI
from pydantic import BaseModel
import base64
from typing import Union
from fastapi.responses import JSONResponse

import controller

class Item(BaseModel):
    image: str

class Output(BaseModel):
    image: str
    text: str

app = FastAPI()


@app.post("/")
def input(item: Item) -> Output:
    situation, base64_string = controller.run(item.image)
    return Output(image=base64_string, text=situation)