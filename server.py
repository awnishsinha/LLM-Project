import asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os
from llama_index.core import StorageContext, load_index_from_storage


load_dotenv()

# Create a RAG tool using LlamaIndex
documents = SimpleDirectoryReader("data").load_data() #loading data from the data directory
index = VectorStoreIndex.from_documents(documents) # creating an index from the documents
#storing the index in the storage context for later retrieval
index.storage_context.persist("storage")
# Later, we can load the index from storage and create a query engine
storage_context = StorageContext.from_defaults(persist_dir="storage")
#loading the index from storage and creating a query engine to query the index
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine(response_mode="tree_summarize") # creating a query engine from the index

# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b

async def search_documents(query: str) -> str:
    """Useful for answering natural language questions about an Affidavit document."""
    response = await query_engine.aquery(query)
    return str(response)

# Create an agent workflow with our calculator tool
agent = FunctionAgent(
    tools=[multiply, search_documents],#function name is passed 
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="""You are a helpful assistant that can perform calculations
    and search through documents to answer questions.""",)


# Now we can ask questions about the documents or do calculations
async def main():
    response = await agent.run(
        "What specific representations did Ampersand7 Pte Ltd make to the Claimants regarding the duration and renewal rights of the “Chagee” sub-franchise agreements, and why do the Claimants allege these representations were false?"
    )
    print(response)

    # Run the agent
if __name__ == "__main__":
    asyncio.run(main())