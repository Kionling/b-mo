import requests
from dotenv import load_dotenv
import os
import openai

# Load environment variables from the .env file
load_dotenv()

# Get the OPENAI_API_KEY environment variable
api_key = os.environ.get('OPENAI_API_KEY')
if api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable not set")
print(api_key)

# Instantiate the OpenAI client with the API key
client = openai.OpenAI(api_key=api_key)

# Make a completion request
completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {'role':'system', 'content':'You are an assistant that speaks like b-mo from adventure time. Short, sweet, and to the point.'}, 
    {"role": "user", "content": sampleContent},
  ]
)

# Print the response
print(completion.choices[0].message)
