from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# import joblib


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

#get audio data from youtube videos
@app.get("/audio")
def get_audio(*args, **kwargs):
   #run pipline_1
    pass

# transcription
@app.get("/transcribe")
def transcribe(*args, **kwargs):
    #run pipline_2
    pass

# summarization
@app.get("/summarize")
def summarize(*args, **kwargs):
    #run pipline_2
    pass

# get keywords from summarization
@app.get("/keywords")
def get_keywords(*args, **kwargs):
    # run pipline_2
    pass
