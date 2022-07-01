FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

RUN \
    apt-get update && \
    apt-get install -y gcc &&\
    apt-get install -y g++

COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY . .

CMD python app.py