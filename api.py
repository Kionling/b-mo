import requests
import os 


def query_openai(prompt):
    api_key = '$OPENAI_API_KEY'
    print(api_key)
    headers = {
        'Authorization': f'Bearer {api_key}',
    }
    data = {
        'model': 'gpt-3.5-turbo',
        'prompt': prompt,
        'max_tokens': 100,
    }

    response = requests.post('https://api.openai.com/v1/engines/davinci/completions', headers=headers, data=data)
    return response.json()
response = query_openai("Hello, my name is John. I am a chatbot created by OpenAI. I am very smart and can answer any question you ask me. I am here to help you with your homework. What is the capital of the United States?")
print(response)
