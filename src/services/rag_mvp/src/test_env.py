# test_env.py
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
print(f"API Key loaded: {api_key[:10]}..." if api_key else "API Key NOT loaded!")
print(f"Key length: {len(api_key) if api_key else 0}")
