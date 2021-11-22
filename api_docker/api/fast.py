from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import youtube_dl
import os
import librosa
import soundfile
from pydub import AudioSegment
import os
from deepspeech import Model
from scipy.io.wavfile import read as wav_read
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from nemo.collections.nlp.models import PunctuationCapitalizationModel
import torch


app = FastAPI()

# params for deepspeech
model_file_path = 'deepspeech-0.9.3-models.pbmm'
lm_file_path = 'deepspeech-0.9.3-models.scorer'
beam_width = 100
lm_alpha = 0.93
lm_beta = 1.18


#CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


#define a root '/' endpoint
@app.get("/")
def index():
    # ⚠️ TODO: get model from GCP
    # pipeline = get_model_from_gcp()
    # pipeline = joblib.load('model.joblib')
    return {"ok": True}


#download youtube video
@app.post("/download")
def get_tube(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        video_title = info_dict.get('id', None)

    path = f'{video_title}.mp3'

    ydl_opts.update({'outtmpl':path})

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return path


#extract audio data from youtube video
@app.get("/extract")
def get_audio(url):
    path =  f"/content/{get_tube(url)}"
    shortcut = path[:-4]
    path_wav = f"{shortcut}.wav"

    sound = AudioSegment.from_file(path)
    sound.export(path_wav, format="wav")

    os.remove(path)

    audio, sr = librosa.load(path_wav, sr=16000)
    soundfile.write(path_wav, data = audio, samplerate = sr)

    return path_wav


# transcription
@app.get("/transcribe")
def get_transcript(url):
    model = Model(model_file_path)
    model.enableExternalScorer(lm_file_path)
    model.setScorerAlphaBeta(lm_alpha, lm_beta)
    model.setBeamWidth(beam_width)

    path_wav = get_audio(url)
    rate, buffer= wav_read(path_wav)
    transcript = model.stt(buffer)

    os.remove(path_wav)

    return transcript


# summarization
@app.get("/summarize")
def summarize(url):
    transcript = get_transcript(url)

    PunctuationCapitalizationModel.list_available_models()
    model = PunctuationCapitalizationModel.from_pretrained("punctuation_en_bert")
    transcript = model.add_punctuation_capitalization([transcript])

    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    device = torch.device('cpu')


    t5_prepared_Text = "summarize: "+transcript[0]
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
    summary_ids = model.generate(tokenized_text,
                                      num_beams=4,
                                      no_repeat_ngram_size=2,
                                      min_length=30,
                                      max_length=100,
                                      early_stopping=True)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return print("\n\nSummarized text: \n",output)


# # get keywords from summary
# @app.get("/keywords")
# def get_keywords(*args, **kwargs):
#     pass
