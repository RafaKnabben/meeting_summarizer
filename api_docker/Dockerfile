# base image
FROM python:3.8.6-buster

# copy api
COPY api /api

# copy package
# COPY audio_summarizer /audio_summarizer

# copy trained model(s)
# COPY model /model

# copy package dependencies
COPY requirements.txt /requirements.txt

# run pip
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
  libsndfile1

# run uvicorn server
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
