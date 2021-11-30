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

# download pre-trained deepspeech
RUN wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
RUN wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
RUN wget https://api.ngc.nvidia.com/v2/models/nvidia/nemo/punctuation_en_bert/versions/1.0.0rc1/files/punctuation_en_bert.nemo

# run uvicorn server
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
