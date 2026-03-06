import os
import time

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
model_name = os.getenv("LLM_MODEL", "gemma-3-27b-it")

genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name=model_name)

print(f"Testing LLM latency for model: {model_name}")
start = time.time()
response = model.generate_content("Hello, this is a latency test. Please respond with 'OK'.")
end = time.time()

print(f"Response: {response.text}")
print(f"Latency: {end - start:.2f} seconds")
