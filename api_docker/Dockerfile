# base image
FROM python:3.8-buster

#api
COPY api /api

# package ???
COPY meeting_summarizer /meeting_summarizer

# models ???
COPY models /models

# requirements
COPY requirements.txt /requirements.txt

# run pip
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# run uvicorn
CMD uvicorn api.fast:app --host 0.0.0.0
#??? --port $PORT
