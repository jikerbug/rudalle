FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

RUN \
    apt-get update && \
    apt-get install -y gcc &&\
    apt-get install -y g++

COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001", "--reload"]

# uvicorn main:app --host 0.0.0.0 --port 5001 --reload