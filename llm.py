from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

response = OpenAI().complete("William Shakespeare is ") # this is getting a completion from the OpenAI LLM using the prompt "William Shakespeare is "
print(response)

handle = OpenAI().stream_complete("William Shakespeare is ",) # this is getting a streaming completion from the OpenAI LLM using the prompt "Write a short biography of William Shakespeare."


for token in handle:
    print(token.delta, end="", flush=True)
