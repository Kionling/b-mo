from gtts import gTTS
import os
import torch
from pydub import AudioSegment
import torchaudio
import torchaudio.transforms as T
from torchaudio.functional import resample

def load_model(model_path):

    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

def preprocess_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    return waveform, sample_rate

def text_to_speech(text):
    speech = gTTS(text)
    speech_file = "speech.mp3"
    speech.save(speech_file)
    os.system('afplay ' + speech_file)
    os.remove(speech_file)
text_to_speech('Hi there!')

