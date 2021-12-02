# base image
FROM python:3.8.6-buster

# copy api
COPY api /api

# copy requirements.txt
COPY requirements.txt /requirements.txt

# run pip
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# install soundfile dependency
RUN apt-get update && apt-get install -y libsndfile1

# install youtube_dl dependency
RUN apt-get update && apt-get install -y ffmpeg

# run uvicorn server
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
