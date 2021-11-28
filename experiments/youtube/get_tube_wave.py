from deepspeech import Model
from scipy.io.wavfile import read as wav_read
from pytube import YouTube
import os
import ffmpeg

def get_tube_wave(url):
    text = (f'{url}')
    yt = YouTube(text)
    stream_url = yt.streams.all()[0].url
    audio, err = (ffmpeg.input(stream_url).output(
        "pipe:", format='wav', acodec='pcm_s16le').run(capture_stdout=True))
    with open('audio.wav', 'wb') as f:
        f.write(audio)
