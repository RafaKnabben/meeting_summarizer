# import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# define a root '/' endpoint
@app.get("/")
def index():

    # ⚠️ TODO: get model from GCP
    # pipeline = get_model_from_gcp()
    # pipeline = joblib.load('model.joblib')

    return {"ok": True}

# transcription
@app.get("/transcribe")
def transcribe(*args, **kwargs):
    #run pipline
    pass

# summarization
@app.get("/summarize")
def summarize(*args, **kwargs):
    #run pipline
    pass

# get keywords
@app.get("/keywords")
def get_keywords(*args, **kwargs):
    # generate keywords from summarization
    pass
