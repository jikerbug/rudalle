import logging
import asyncio
from fastapi import FastAPI, HTTPException

import config
import model

app = FastAPI()

# logger setting
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

APIRequest = config.APIRequest
APIResponse = config.APIResponse

@app.on_event("startup")
async def startup_event():
    logging.info("start server")

@app.post("/generate")
async def generate(request: APIRequest)->APIResponse:

    req_data = request.dict()
    
    args = []
    text_input = req_data["text"]
    num_images = req_data["num_images"]
    args.append(text_input)
    args.append(num_images)

    logger.info(f"input: {text_input}")
    
    req = {'input': args}
    output = await asyncio.create_task(make_images(req['input'][0], req['input'][1]))
    logger.info(f"output: {output}")

    # output check
    if "error" in output:
        raise HTTPException(status_code=500, detail=f"Internal server error: {output['error']}")
    else:
        result = output["result"]
        return APIResponse(text=result)

@app.get("/healthz", status_code=200)
def check_health():
    return "healthy"