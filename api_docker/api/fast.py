from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import youtube_dl
import os
import librosa
import soundfile
from pydub import AudioSegment
from deepspeech import Model
from scipy.io.wavfile import read as wav_read
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import nemo
import nemo.collections.nlp as nemo_nlp
import torch

app = FastAPI()

# params for Deepspeech
model_file_path = './deepspeech-0.9.3-models.pbmm'
lm_file_path = './deepspeech-0.9.3-models.scorer'
beam_width = 100
lm_alpha = 0.93
lm_beta = 1.18

# params for T5
num_beams = 4
no_repeat_ngram_size = 2
min_length = 30
max_length = 100
early_stopping = True

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# functions
# download youtube video
def get_tube(url):
    ydl_opts = {
        'format':
        'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        video_title = info_dict.get('id', None)

    mp3 = f'{video_title}.mp3'
    ydl_opts.update({'outtmpl': mp3})

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return mp3


# extract audio data from youtube video
def get_audio(mp3):
    shortcut = mp3[:-4]
    wav = f"{shortcut}.wav"

    sound = AudioSegment.from_file(mp3)
    sound.export(wav, format="wav")

    os.remove(mp3)

    audio, sr = librosa.load(wav, sr=16000)
    soundfile.write(wav, data=audio, samplerate=sr)

    return wav


# transcription
def get_transcript(wav):
    model = Model(model_file_path)
    model.enableExternalScorer(lm_file_path)
    model.setScorerAlphaBeta(lm_alpha, lm_beta)
    model.setBeamWidth(beam_width)

    rate, buffer = wav_read(wav)
    transcript = model.stt(buffer)

    os.remove(wav)

    return transcript


# summarization
def get_summary(transcript):
    model_1 = nemo_nlp.models.PunctuationCapitalizationModel.from_pretrained(
        model_name="punctuation_en_bert")
    transcript = model_1.add_punctuation_capitalization([transcript])

    model_2 = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    device = torch.device('cpu')

    t5_prepared_Text = "summarize: " + transcript[0]
    tokenized_text = tokenizer.encode(t5_prepared_Text,
                                      return_tensors="pt").to(device)
    summary_ids = model_2.generate(tokenized_text,
                                   num_beams=num_beams,
                                   no_repeat_ngram_size=no_repeat_ngram_size,
                                   min_length=min_length,
                                   max_length=max_length,
                                   early_stopping=early_stopping)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summary = {"Summarized text": output}

    return summary


# endpoints
# define a root '/'
@app.get("/")
def index():
    return {"ok": True}


@app.post("/download_test")
def get_tube_only(url):
    mp3 = get_tube(url)
    return mp3


@app.get("/extract_test")
def get_audio_only(url):
    mp3 = get_tube_only(url)
    wav = get_audio(mp3)
    return wav


@app.get("/transcribe_test")
def get_transcript_only(url):
    wav = get_audio_only(url)
    transcript = get_transcript(wav)
    return transcript


@app.get("/summarize_test")
def get_summary_only(url):
    transcript = get_transcript_only(url)
    summary = get_summary(transcript)
    return summary


@app.get("/all_steps_test")
def get_all(url):
    mp3 = get_tube(url)
    wav = get_audio(mp3)
    transcript = get_transcript(wav)
    summary = get_summary(transcript)
    return summary


# # get keywords from summary
# @app.get("/keywords")
# def get_keywords(*args, **kwargs):
#     pass
