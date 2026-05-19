from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

response = OpenAI().complete("William Shakespeare is ") # this is getting a completion from the OpenAI LLM using the prompt "William Shakespeare is "
print(response)


