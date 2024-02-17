from gtts import gTTS
import os
import torch

def text_to_speech(text):
    speech = gTTS(text)
    speech_file = "speech.mp3"
    speech.save(speech_file)
    os.system('afplay ' + speech_file)
    os.remove(speech_file)
text_to_speech('Hi there!')

# Assuming your model is a PyTorch model
model = torch.load('path/to/your/model.pth')

# Make sure to set the model to evaluation mode
model.eval()
