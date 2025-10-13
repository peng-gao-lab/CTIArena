from openai import OpenAI
from dotenv import load_dotenv
import time
import requests
import os
import random


try:
    from google import genai
    from google.genai.types import HttpOptions
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("Warning: google-genai not available. Gemini models will not work.")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic not available. Claude models will not work.")

load_dotenv()


google_api_key = os.getenv("GOOGLE_API_KEY")
claude_api_key = os.getenv("CLAUDE_API_KEY")


if GOOGLE_AVAILABLE:
    google_client = genai.Client(http_options=HttpOptions(api_version="v1"))

if ANTHROPIC_AVAILABLE:
    Anthropic_client = anthropic.Anthropic(
        api_key=claude_api_key,
    )

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1",
)

model_mapping = {
    
    'llama3-405b': 'llama3.1:405b',
    'llama3-8b': 'llama3.1:8b',

    
    'qwen3-235b': 'qwen3:235b',

    
    'phi4': 'phi4:14b',

    
    'gpt4o': 'gpt-4o',
    'gpt5': 'gpt-5',
    'gemini-2.5-pro': 'gemini-2.5-pro',
    'gemini-2.5-flash': 'gemini-2.5-flash',
    'claude-sonnet-4': 'claude-sonnet-4-20250514',
    'claude-3-5-haiku': 'claude-3-5-haiku-20241022'
    
}


temperature = 0.7
top_p = 0.8
seed = 42
max_tokens = 4096

###### before use this function, you need config the parameters, spcific for different models

sys_prompt = 'You are a cybersecurity expert specializing in cyberthreat intelligence.'

def get_single_prediction(model_name, question):
    
    if model_name.startswith('llama3') or model_name.startswith('deepseek') or model_name.startswith('qwen3'):


        model = model_mapping[model_name]
        prompt = sys_prompt + ' ' + question

        
        attempt = 0
        while True:
            try:
        
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_predict": 4096 
                    }
                )
                output = response.json()['response']
                break   
            except Exception as e:
                attempt += 1
                wait_time = min(60, (2 ** attempt) + random.random())  
                print(f"Warning: Gemini API call failed (attempt {attempt}): {e}")
                print(f"Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)

    elif model_name in ["mistral", "phi4"]:

        model_id = model_mapping[model_name]
        prompt = sys_prompt + ' ' + question

        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_id,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": -1 
            }
        )

        output = response.json()['response']

    elif model_name.startswith('gpt'):

        
        if  model_name in ['gpt5', 'gpt_o3_mini', 'gpt_o4_mini']: 
            model = model_mapping[model_name]
            response = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': sys_prompt},
                    {'role': 'user', 'content': question}
                ],
                seed = seed
            )
        else:
            model = model_mapping[model_name]
            response = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': sys_prompt},
                    {'role': 'user', 'content': question}
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed
            )
        output = response.choices[0].message.content

    elif model_name.startswith('gemini'):
        
        if not GOOGLE_AVAILABLE:
            return "Error: google-genai not available. Please install it to use Gemini models."

        model = model_mapping[model_name]
        prompt = sys_prompt + ' ' + question

        
        attempt = 0
        while True:
            try:
                response = google_client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                output = response.text
                break   
            except Exception as e:
                attempt += 1
                wait_time = min(60, (2 ** attempt) + random.random())  
                print(f"Warning: Gemini API call failed (attempt {attempt}): {e}")
                print(f"Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)

        
        time.sleep(3)

    elif model_name.startswith('claude'):
        
        if not ANTHROPIC_AVAILABLE:
            return "Error: anthropic not available. Please install it to use Claude models."
        model = model_mapping[model_name]
        prompt = sys_prompt + ' ' + question
        response = Anthropic_client.messages.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens
        )
        output = response.content[0].text

    return output

if __name__ == '__main__':

    print(get_single_prediction('mistral', "hello"))