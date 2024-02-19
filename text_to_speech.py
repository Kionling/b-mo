from gtts import gTTS
import os

def text_to_speech(text):
    speech = gTTS(text)
    speech_file = "speech.mp3"
    speech.save(speech_file)
    os.system('afplay ' + speech_file)

text_to_speech('Hello, world!') 