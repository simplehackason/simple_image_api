from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

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

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
def input(item: Item) -> Data:
    situation, base64_string, object_names = controller.run(item.image)
    return Data(
        data=Output(
            image=base64_string, description=situation, object_names=object_names
        )
    )
