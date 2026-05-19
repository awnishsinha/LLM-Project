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

def load_or_create_index():
    if os.path.exists("storage"):
        print("Loading index from storage...")
        storage_context = StorageContext.from_defaults(persist_dir="storage") # creating a storage context from the persisted index in the storage directory
        index = load_index_from_storage(storage_context) # loading the index from storage if it exists
    else:
        print("Creating new index...")
        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents) # creating an index from the documents
        index.storage_context.persist("storage") # persisting the index to storage for future use
    return index


index=load_or_create_index()

# We can use the SentenceSplitter to split the documents into sentences and create nodes from them
sentence=SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=20
)

# nodes=sentence.get_nodes_from_documents(documents) # splitting the documents into sentences and creating nodes from them
# print("Total chunks:", len(nodes))

# for i, node in enumerate(nodes[:5]):
#     print("\n======================")
#     print("CHUNK:", i + 1)
#     print("======================")
#     print(node.text[:1000])
    

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
    ctx=Context(agent)
    
    response1 = await agent.run(
        user_msg="Hi, my name is Awnish.",
        ctx=ctx
    )
    print("Response 1:")
    print(response1)

    response2 = await agent.run(
        user_msg="What is my name?",
        ctx=ctx
    )
    print("\nResponse 2:")
    print(response2)

    response3 = await agent.run(
        user_msg="Why financial losses happened in the case?",
        ctx=ctx
    )
    print("\nResponse 3:")
    print(response3)

    response4 = await agent.run(
        user_msg="What is 1234 * 4567?",
        ctx=ctx
    )
    print("\nResponse 4:")
    print(response4)

    # Run the agent
if __name__ == "__main__":
    asyncio.run(main())