from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pytube import YouTube
import ffmpeg
from summarizer import Summarizer
import os
from deepspeech import Model
from scipy.io.wavfile import read as wav_read
from transformers import AutoTokenizer, AutoModelWithLMHead, T5Tokenizer, T5ForConditionalGeneration, T5Config
import nemo
import nemo.collections.nlp as nemo_nlp
import torch


app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


def get_wave(url):
    text = (f'{url}')
    yt = YouTube(text)
    stream_url = yt.streams.all()[0].url
    audio, err = (ffmpeg.input(stream_url).output(
        "pipe:", format='wav', acodec='pcm_s16le').run(capture_stdout=True))
    with open('audio.wav', 'wb') as f:
        f.write(audio)
    audio_file = 'audio.wav'

    return audio_file


def get_row_text(audio_file):
    model = Model("./deepspeech-0.9.3-models.pbmm")
    model.enableExternalScorer("./deepspeech-0.9.3-models.scorer")
    model.setBeamWidth(3000)
    rate, buffer = wav_read(audio_file)
    row_text = model.stt(buffer)

    os.remove(audio_file)

    return row_text


def get_row_sentence(row_text):
    model = nemo_nlp.models.PunctuationCapitalizationModel.from_pretrained(
        model_name="punctuation_en_distilbert")
    row_sentence = model.add_punctuation_capitalization([row_text])

    return row_sentence


def get_corr_text(row_sentence):
    tokenizer = AutoTokenizer.from_pretrained(
        "flexudy/t5-base-multi-sentence-doctor")
    model = AutoModelWithLMHead.from_pretrained(
        "flexudy/t5-base-multi-sentence-doctor")
    input_ids = tokenizer.encode(row_sentence, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=32, num_beams=1)
    text = tokenizer.decode(outputs[0],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True)

    assert text


def get_abs_summary(text):
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    device = torch.device('cpu')

    t5_prepared_Text = "summarize: " + text[0]
    tokenized_text = tokenizer.encode(t5_prepared_Text,
                                      return_tensors="pt").to(device)
    summary_ids = model.generate(tokenized_text,
                                 num_beams=4,
                                 no_repeat_ngram_size=2,
                                 min_length=30,
                                 max_length=100,
                                 early_stopping=True)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    abs_summary = {"Summarized text": output}

    return abs_summary["Summarized text"]


def get_ext_summary(text):
    ext_sumary = Summarizer(text[0])

    return get_ext_summary



# endpoints
@app.get("/")
def index():
    return {"ok": True}


@app.post("/download_test")
def get_wave_only(url):
    wave = get_wave(url)
    return wave


@app.get("/extract_test")
def get_row_text_only(url):
    wave = get_wave_only(url)
    row_text = get_row_text(wave)
    return row_text


@app.get("/compose_test")
def get_row_sentence_only(url):
    row_text= get_row_text_only(url)
    row_sentence = get_row_sentence(row_text)
    return row_sentence


@app.get("/correct_test")
def get_corr_text_only(url):
    row_sentence = get_row_sentence_only(url)
    text = get_corr_text(row_sentence)
    return text


@app.get("/abs_summarize_test")
def get_abs_summ_only(url):
    text = get_corr_text_only(url)
    abs_summary = get_abs_summary(text)
    return abs_summary


@app.get("ext_summarize_test")
def get_ext_summ_only(url):
    text = get_corr_text_only(url)
    ext_summary = get_ext_summary(text)
    return ext_summary


@app.get("/abs_summ_all")
def get_abs_all(url):
    wave = get_wave(url)
    row_text = get_row_text(wave)
    row_sentence = get_row_sentence(row_text)
    text = get_corr_text(row_text)
    abs_summary = get_abs_summary(text)
    return abs_summary


@app.get("/ext_summ_all")
def get_ext_all(url):
    wave = get_wave(url)
    row_text = get_row_text(wave)
    row_sentence = get_row_sentence(row_text)
    text = get_corr_text(row_text)
    ext_summary = get_ext_summary(text)
    return ext_summary
