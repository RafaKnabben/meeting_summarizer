import numpy as np
import os
import wave
import librosa
import soundfile
import pyaudio
import io
import ffmpeg

from deepspeech import Model
from IPython.display import Audio
from IPython.display import HTML, Audio
from google.colab.output import eval_js
from base64 import b64decode
from scipy.io.wavfile import read as wav_read

# Important files to run the DeepSpeech model:
# !curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
# !curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
# Move these files to: /meeting_summarizer/speech2vec-models/data

# Here's the data to train the model
# !curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/audio-0.9.3.tar.gz
# In order to open the file above, use the following command:
# !tar xvf audio-0.9.3.tar.gz

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

    rate, buffer= wav_read(audio_file)
    return model.stt(buffer)


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
recordButton.addEventListener("click", toggleRecording);
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
  
    process = (ffmpeg
        .input('pipe:0')
        .output('pipe:1', format='wav')
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True, overwrite_output=True))
    
    output, err = process.communicate(input=binary)
  
    riff_chunk_size = len(output) - 8

    q = riff_chunk_size
    b = []
    for i in range(4):
        q, r = divmod(q, 256)
        b.append(r)
    riff = output[:4] + bytes(b) + output[8:]

    rate, buffer = wav_read(io.BytesIO(riff))

    return buffer, rate

def get_audio():
    buffer, rate = get_mic()
    soundfile.write("/data/mic_result.wav", data = buffer, samplerate = rate)

    if rate != 16000:
        audio, sr = librosa.load('/data/mic_result.wav', sr=16000)
        soundfile.write("/data/mic_result.wav", data = audio, samplerate = sr)
    
    transcript = get_transcript("/data/mic_result.wav")

    return transcript
    
    else:
        transcript = get_transcript("/data/mic_result.wav")
    
    return transcript

get_audio()
