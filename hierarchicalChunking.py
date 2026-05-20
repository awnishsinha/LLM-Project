from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core import VectorStoreIndex

node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]
)

nodes = node_parser.get_nodes_from_documents(documents)

index = VectorStoreIndex(nodes)