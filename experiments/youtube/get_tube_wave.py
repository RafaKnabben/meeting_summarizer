from deepspeech import Model
from scipy.io.wavfile import read as wav_read
from pytube import YouTube
import os
import ffmpeg

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


def get_text(audio_file):
    model_file_path = "./deepspeech-0.9.3-models.pbmm"
    lm_file_path = "./deepspeech-0.9.3-models.scorer"
    beam_width = 3000
    lm_alpha = 0.931289039105002
    lm_beta = 1.1834137581510284

    model = Model(model_file_path)
    model.enableExternalScorer(lm_file_path)
    model.setScorerAlphaBeta(lm_alpha, lm_beta)
    model.setBeamWidth(beam_width)
    rate, buffer = wav_read(audio_file)
    row_text = model.stt(buffer)

    return row_text


def get_sentence(unpunctuated_text):
    model = PunctuationCapitalizationModel.from_pretrained(
        model_name="punctuation_en_distilbert")
    punctuated_text = model.add_punctuation_capitalization([unpunctuated_text])

    return punctuated_text

def get_summary(text):
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    device = torch.device('cpu')

    t5_prepared_Text = "summarize: " + transcript[0]
    tokenized_text = tokenizer.encode(t5_prepared_Text,
                                      return_tensors="pt").to(device)
    summary_ids = model.generate(tokenized_text,
                                 num_beams=4,
                                 no_repeat_ngram_size=2,
                                 min_length=30,
                                 max_length=100,
                                 early_stopping=True)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return print("\n\nSummarized text: \n", output)
