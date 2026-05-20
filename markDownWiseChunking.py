from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()

parser = MarkdownNodeParser()
nodes = parser.get_nodes_from_documents(documents)

index = VectorStoreIndex(nodes)