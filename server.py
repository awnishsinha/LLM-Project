import os
import asyncio
import hashlib
from llama_index.core import PromptTemplate
import dotenv
from pathlib import Path

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI



from pageWiseChunking import load_pdf_page_wise

dotenv.load_dotenv()

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "docusearch_documents"


def get_file_id(file_path: Path) -> str:
    file_path = file_path.resolve()
    raw = f"{file_path}|{file_path.stat().st_size}|{file_path.stat().st_mtime}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def get_chroma_collection():
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    return db.get_or_create_collection(COLLECTION_NAME)


def get_index_from_chroma():
    chroma_collection = get_chroma_collection()

    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store
    )


def get_pdf_files(source_path: str):
    path = Path(source_path)

    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Only PDF files are supported currently: {path}")
        return [path]

    if path.is_dir():
        return list(path.rglob("*.pdf"))

    return []


def index_selected_path(source_path: str):
    pdf_files = get_pdf_files(source_path)

    if not pdf_files:
        raise ValueError("No PDF files found in selected path.")

    chroma_collection = get_chroma_collection()

    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    splitter = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=20
    )

    all_nodes = []

    for pdf_file in pdf_files:
        file_id = get_file_id(pdf_file)

        existing = chroma_collection.get(
            where={"file_id": file_id},
            limit=1
        )

        if existing and existing.get("ids"):
            print(f"Already indexed, skipping: {pdf_file.name}")
            continue

        print(f"Indexing file: {pdf_file}")

        documents = load_pdf_page_wise(str(pdf_file))
        nodes = splitter.get_nodes_from_documents(documents)

        for i, node in enumerate(nodes):
            node.id_ = f"{file_id}_chunk_{i}"

            node.metadata.update({
                "file_id": file_id,
                "file_name": pdf_file.name,
                "file_path": str(pdf_file),
                "source_path": str(source_path),
                "chunk_no": i,
            })

        all_nodes.extend(nodes)

    if not all_nodes:
        print("No new files to index.")
        return get_index_from_chroma()

    index = VectorStoreIndex(
        all_nodes,
        storage_context=storage_context
    )

    print(f"Indexed new chunks: {len(all_nodes)}")

    return get_index_from_chroma()

index=index_selected_path(r"C:\Users\Xaira\MGC LLC\Test File 2 OC 673\Affidavits")

query_engine=index.as_query_engine();

async def search_document(query: str)-> str:
	"""Useful for answering natural language questions about an Affidavit document."""
	response = await query_engine.aquery(query)
	return str(response)

agent=FunctionAgent(
	tools=[search_document],
	llm=OpenAI(model="gpt-4o-mini"),
	system_prompt="""You are a helpful assistant that can search through documents to answer questions."""
)

async def main():
	question = "Find the key points in the Gan Wan Shan affidavit?"
	response = await agent.run(question)
	print(f"Question: {question}")
	print(f"Answer: {response}")
 
if __name__ == "__main__":
	asyncio.run(main())