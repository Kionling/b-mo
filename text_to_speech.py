from gtts import gTTS
import os
import torch
from pydub import AudioSegment
import torchaudio
import torchaudio.transforms as T
from torchaudio.functional import resample
import torch.optim as optim 
import torch.nn as nn 



class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = TheModelClass()

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])




    
def load_model(model_path):
    # Instantiate your model (replace `YourModelClass` with the actual class of your model)
    model = TheModelClass()
    
    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    for key, value in state_dict.items():
        print(f"{key}: {type(value)}")
    # Load the state dictionary into your model
    model.load_state_dict(state_dict)
    
    # Set the model to evaluation mode
    model.eval()
    
    return model


def preprocess_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    # Add any necessary preprocessing steps here. For example, resampling:
    # waveform = resample(waveform, orig_freq=sample_rate, new_freq=16000)
    return waveform, sample_rate

def apply_model_to_audio(model, waveform):
    with torch.no_grad():
        # Your model's specific invocation may vary
        transformed_waveform = model(waveform)
    return transformed_waveform

def save_waveform_to_mp3(waveform, sample_rate, output_path):
    torchaudio.save(output_path, waveform, sample_rate)
    # Convert the output file to MP3 if necessary, using ffmpeg or similar

def text_to_speech_with_voice_change(text, model):
    # Convert text to speech and save as an MP3
    speech = gTTS(text)
    speech_file = "speech.mp3"
    speech.save(speech_file)
    
    # Load and preprocess the audio
    waveform, sample_rate = preprocess_audio(speech_file)
    
    # Apply the model to transform the voice
    transformed_waveform = apply_model_to_audio(model, waveform)
    
    # Save the transformed waveform to an MP3 file
    transformed_file = "transformed_speech.mp3"
    save_waveform_to_mp3(transformed_waveform, sample_rate, transformed_file)
    
    # Play the transformed speech file
    os.system('afplay ' + transformed_file)
    
    # Cleanup
    os.remove(speech_file)
    os.remove(transformed_file)

# Load your model
model_path = 'models/model.pth'  # Update this path as necessary
model = load_model(model_path)

# Example usage
text_to_speech_with_voice_change('Hi there!', model)
