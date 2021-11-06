from deepspeech import Model
from IPython.display import Audio
import numpy as np
import os
import wave
import librosa
import soundfile
import pyaudio
from IPython.display import HTML, Audio
from js2py import eval_js
from base64 import b64decode
import numpy as np
from scipy.io.wavfile import read as wav_read
import io
import ffmpeg
from IPython import display


def read_wav_file(filename):
    with wave.open(filename, 'rb') as w:
        rate = w.getframerate()
        frames = w.getnframes()
        buffer = w.readframes(frames)
        print("Rate:", rate)
        print("Frames:", frames)
        print("Buffer Len:", len(buffer))

    return buffer, rate


def get_transcript(audio_file):
    model_file_path = "/data/deepspeech-0.9.3-models.pbmm"
    lm_file_path = "/data/deepspeech-0.9.3-models.scorer"
    beam_width = 100
    lm_alpha = 0.93
    lm_beta = 1.18

    model = Model(model_file_path)
    model.enableExternalScorer(lm_file_path)

    model.setScorerAlphaBeta(lm_alpha, lm_beta)
    model.setBeamWidth(beam_width)

    buffer, rate = read_wav_file(audio_file)
    data16 = np.frombuffer(buffer, dtype=np.int16)
    return model.stt(data16)


AUDIO_HTML = """
<script>
var my_div = document.createElement("DIV");
var my_p = document.createElement("P");
var my_btn = document.createElement("BUTTON");
var t = document.createTextNode("Press to start recording");

my_btn.appendChild(t);
//my_p.appendChild(my_btn);
my_div.appendChild(my_btn);
document.body.appendChild(my_div);

var base64data = 0;
var reader;
var recorder, gumStream;
var recordButton = my_btn;

var handleSuccess = function(stream) {
  gumStream = stream;
  var options = {
    //bitsPerSecond: 8000, //chrome seems to ignore, always 48k
    mimeType : 'audio/webm;codecs=opus'
    //mimeType : 'audio/webm;codecs=pcm'
  };
  //recorder = new MediaRecorder(stream, options);
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = function(e) {
    var url = URL.createObjectURL(e.data);
    var preview = document.createElement('audio');
    preview.controls = true;
    preview.src = url;
    document.body.appendChild(preview);

    reader = new FileReader();
    reader.readAsDataURL(e.data);
    reader.onloadend = function() {
      base64data = reader.result;
      //console.log("Inside FileReader:" + base64data);
    }
  };
  recorder.start();
  };

recordButton.innerText = "Recording... press to stop";

navigator.mediaDevices.getUserMedia({audio: true}).then(handleSuccess);


function toggleRecording() {
  if (recorder && recorder.state == "recording") {
      recorder.stop();
      gumStream.getAudioTracks()[0].stop();
      recordButton.innerText = "Saving the recording... pls wait!"
  }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

var data = new Promise(resolve=>{
//recordButton.addEventListener("click", toggleRecording);
recordButton.onclick = ()=>{
toggleRecording()

sleep(2000).then(() => {
  // wait 2000ms for the data to be available...
  // ideally this should use something like await...
  //console.log("Inside data:" + base64data)
  resolve(base64data.toString())

});

}
});

</script>
"""


def get_mic():
    display(HTML(AUDIO_HTML))
    data = eval_js("data")
    binary = b64decode(data.split(',')[1])

    process = (ffmpeg.input('pipe:0').output('pipe:1', format='wav').run_async(
        pipe_stdin=True,
        pipe_stdout=True,
        pipe_stderr=True,
        quiet=True,
        overwrite_output=True))
    output, err = process.communicate(input=binary)

    riff_chunk_size = len(output) - 8
    # Break up the chunk size into four bytes, held in b.
    q = riff_chunk_size
    b = []
    for i in range(4):
        q, r = divmod(q, 256)
        b.append(r)

    # Replace bytes 4:8 in proc.stdout with the actual size of the RIFF chunk.
    riff = output[:4] + bytes(b) + output[8:]

    sr, audio = wav_read(io.BytesIO(riff))

    return audio, sr

def get_audio():
    play, rate = get_mic()
    soundfile.write("/content/mic_result.wav", data=play, samplerate=rate)

    if rate != 16000:
        audio, sr = librosa.load('/content/mic_result.wav', sr=16000)
        soundfile.write("/content/mic_result.wav", data=audio, samplerate=sr)

        transcript = get_transcript("/content/mic_result.wav")

        return transcript

    else:
        transcript = get_transcript("/content/mic_result.wav")

    return transcript

get_audio()
