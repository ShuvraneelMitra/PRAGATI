import os
from smolagents import LiteLLMModel, CodeAgent
from dotenv import load_dotenv
load_dotenv()

messages = [
  {"role": "user", "content": [{"type": "text", "text": "Hello, how are you?"}]}
]

model = LiteLLMModel("gemini/gemini-1.5-flash",
                     api_key=os.getenv("GEMINI_API_KEY"),
                     temperature=0.2, max_tokens=100,)
