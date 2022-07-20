

import sys
sys.path.insert(0, './rudalle-aspect-ratio')
from rudalle_aspect_ratio import RuDalleAspectRatio, get_rudalle_model
from rudalle import get_vae, get_tokenizer
from rudalle.pipelines import show

device = 'cuda'
dalle = get_rudalle_model('Surrealist_XL', fp16=True, device=device)
vae, tokenizer = get_vae().to(device), get_tokenizer()
rudalle_ar = RuDalleAspectRatio(
    dalle=dalle, vae=vae, tokenizer=tokenizer,
    aspect_ratio=32/9, bs=4, device=device
)




'''
    server!
'''

from flask import Flask, request, jsonify

from tqdm import tqdm
import numpy as np
from queue import Queue, Empty
from threading import Thread
import time

# torch

import torch

from einops import repeat

# vision imports

from PIL import Image

import base64
from io import BytesIO

from typing import Union

from fastapi import Request, FastAPI

app = FastAPI()


# generate images

image_size = vae.image_size


async def make_images(text_input, num_images):
    
    print(text_input)
    _, result_pil_images = rudalle_ar.generate_images(text_input, 768, 0.99, 1)

    response = []

    print(result_pil_images)

    for img in result_pil_images:

        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        response.append(img_str)

    return response



@app.post('/generate')
async def generate(request: Request):
    
    json_data = await request.json()


    text_input = json_data["text"]
    num_images = json_data["num_images"]


    if num_images > 10:
        return {'Error': 'Too many images requested. Request no more than 10'}.json(), 500

    args = []
    args.append(text_input)
    args.append(num_images)


    req = {'input': args}
    req["output"] = await make_images(req['input'][0], req['input'][1])

    print(req['output'])
    return req['output']


@app.get("/healthz", status_code=200)
def check_health():
    return "healthy"

