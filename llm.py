from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
from llama_index.core.llms import ChatMessage
import asyncio
load_dotenv()
llm= OpenAI()
# response = llm.complete("William Shakespeare is ") # this is getting a completion from the OpenAI LLM using the prompt "William Shakespeare is "
# print(response)

# handle = llm.stream_complete("William Shakespeare is ",) # this is getting a streaming completion from the OpenAI LLM using the prompt "Write a short biography of William Shakespeare."


# for token in handle:
#     print(token.delta, end="", flush=True)


# ChatInterface

# messages = [
#     ChatMessage(role="system", content="You are a helpful assistant."),
#     ChatMessage(role="user", content="Tell me a joke."),
# ]
# chat_response = llm.chat(messages)

# print(chat_response)

#Synchronous Streaming chat response
# messages = [
#     ChatMessage(role="system", content="You are a helpful assistant."),
#     ChatMessage(role="user", content="Tell me a joke."),
# ]

# stream_chat_handle=llm.stream_chat(messages)
# for token in stream_chat_handle:
#     print(token.delta, end="", flush=True)
    

# Asynchronous Streaming chat response

messages=[
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(role="user", content="Tell me a joke."),
]

async def main():
    stream_chat_handle= await llm.astream_chat(messages)

    async for token in stream_chat_handle:
        print(token.delta, end="", flush=True)
        

if __name__ == "__main__":
    asyncio.run(main())        