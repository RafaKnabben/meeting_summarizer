# base image
FROM python:3.8.6-buster

# copy api
COPY api2 /api2

# copy requirements.txt
COPY requirements.txt /requirements.txt

# run pip
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# install dependencies
RUN apt-get update && apt-get install -y libsndfile1
RUN apt-get install -y ffmpeg

# download pre-trained deepspeech
RUN wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
RUN wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer

# run uvicorn server
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
