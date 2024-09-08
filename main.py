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
    description: str
    object_names: list[str]

class Data(BaseModel):
    data: Output

app = FastAPI()


@app.post("/")
def input(item: Item) -> Data:
    situation, base64_string, object_names = controller.run(item.image)
    return Data(image=base64_string, description=situation, object_names=object_names)