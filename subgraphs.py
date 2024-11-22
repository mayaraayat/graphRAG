from typing import List, Optional
from langchain.schema import Document
from langchain_community.graphs.graph_document import GraphDocument
from entities_extraction import get_extraction_chain
from utils import map_to_base_node, map_to_base_relationship


def extract_graph(
    document: Document,
    nodes: Optional[List[str]] = None,
    rels: Optional[List[str]] = None,
) -> None:
    # Extract graph data using OpenAI functions
    extract_chain = get_extraction_chain(nodes, rels)
    try:
        data = extract_chain.invoke(document.page_content)["function"]
    except Exception as e:
        raise ValueError(f"Extraction failed: {e}")
    # Construct a graph document

    graph_document = GraphDocument(
        nodes=[map_to_base_node(node) for node in data.nodes],
        relationships=[map_to_base_relationship(rel) for rel in data.rels],
        source=document,
    )

    return graph_document
