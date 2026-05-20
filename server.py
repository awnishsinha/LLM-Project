import asyncio
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.workflow import Context
from llama_index.core.node_parser import  SentenceSplitter
from networkx import nodes
from pageWiseChunking import load_pdf_page_wise
from llama_index.vector_stores.chroma import ChromaVectorStore


load_dotenv()

def load_or_create_index():
	if os.path.exists("storag"):
		print("Loading index from storage...")
		storage_context = StorageContext.from_defaults(persist_dir="storage") # creating a storage context from the persisted index in the storage directory
		index = load_index_from_storage(storage_context) # loading the index from storage if it exists
	else:
		print("Creating new index...")
		documents = load_pdf_page_wise(r"C:\Users\Xaira\MGC LLC\Test File 2 OC 673\Affidavits\25 09 08 1st Affidavit of Gan Wan Shan.pdf")  # Load PDF documents page-wise
   
		# initialize client, setting path to save data
		db = chromadb.PersistentClient(path="./chroma_db")
		# create collection
		chroma_collection = db.get_or_create_collection("quickstart")

		# assign chroma as the vector_store to the context
		vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
		storage_context = StorageContext.from_defaults(vector_store=vector_store)
		index = VectorStoreIndex.from_documents(documents, storage_context=storage_context) # creating a new index from the documents and the storage context
	return index


index=load_or_create_index()




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