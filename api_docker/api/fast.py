'''
need to be improved !!!!!
'''


from fastapi import FastAPI

app = FastAPI()


# define a root '/' endpoint
@app.get("/")
def index():
    # load pipeline
    return {"ok": True}


# make speech recognition (transcription) & summarization
@app.get("/summarize")
def summarize(*args, **kwargs):
    # run pipline
    pass


# get keywords
@app.get("/keywords")
def get_keywords(*args, **kwargs):
    # generate keywords from transcription/summarization
    pass
