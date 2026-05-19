import asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.workflow import Context
from llama_index.core.node_parser import  SentenceSplitter


load_dotenv()

# Create a RAG tool using LlamaIndex
documents = SimpleDirectoryReader("data").load_data() #loading data from the data directory

sentence=SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=20
)

nodes=sentence.get_nodes_from_documents(documents) # splitting the documents into sentences and creating nodes from them
print("Total chunks:", len(nodes))

for i, node in enumerate(nodes[:5]):
    print("\n======================")
    print("CHUNK:", i + 1)
    print("======================")
    print(node.text[:1000])
    
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

ctx=Context(agent)

# Now we can ask questions about the documents or do calculations
async def main():
    response = await agent.run(
        "Why financial losses happened in the case?",
        ctx=ctx
    )
    print(response)

    # Run the agent
if __name__ == "__main__":
    asyncio.run(main())