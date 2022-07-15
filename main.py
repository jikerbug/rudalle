

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

from fastapi import FastAPI

app = FastAPI()


# load model

batch_size = 4

top_k = 0.9

# generate images

image_size = vae.image_size



async def make_images(text_input, num_images):
    try:
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

    except Exception as e:
        print('Error occur in script generating!', e)
        return jsonify({'Error': e}), 500


@app.post('/generate')
async def generate():
    try:
        args = []
        json_data = request.get_json()
        text_input = json_data["text"]
        num_images = json_data["num_images"]

        if num_images > 10:
            return jsonify({'Error': 'Too many images requested. Request no more than 10'}), 500

        args.append(text_input)
        args.append(num_images)

    except Exception as e:
        return jsonify({'Error': 'Invalid request'}), 500

    req = {'input': args}
    req["output"] = await make_images(req['input'][0], req['input'][1])

    return jsonify(req['output'])


@app.route('/healthz', methods=["GET"])
def health_check():
    return "Health", 200

