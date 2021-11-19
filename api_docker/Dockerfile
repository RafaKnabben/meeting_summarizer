# base image
FROM python:3.8-buster

# path to the GCP credentials file
# COPY /path/to/your/credentials.json /credentials.json

# copy api
COPY api /api

# copy package
# COPY meeting_summarizer /meeting_summarizer

# copy trained model(s)
# COPY model /model

# copy package dependencies
COPY requirements.txt /requirements.txt

# run pip
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# run uvicorn server
CMD uvicorn api.fast:app --host 0.0.0.0
#??? --port $PORT
