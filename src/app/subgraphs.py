from langchain.schema import Document
from langchain_community.graphs.graph_document import GraphDocument
from src.app.entities_extraction import get_extraction_chain
from src.app.utils.utils import map_to_base_node, map_to_base_relationship


def extract_graph(
    document: Document,
    nodes: list[str] | None = None,
    rels: list[str] | None = None,
) -> GraphDocument:
    """
    Extract graph data from a document and construct a GraphDocument.

    Args:
        document (Document): The input document to extract data from.
        nodes (list[str] | None): List of node types to extract (default is None).
        rels (list[str] | None): List of relationship types to extract (default is None).

    Returns:
        GraphDocument: A graph representation of the extracted data.

    Raises:
        ValueError: If the extraction process fails.
    """
    # Get the extraction chain based on specified nodes and relationships
    extract_chain = get_extraction_chain(nodes, rels)

    try:
        extracted_data = extract_chain.invoke(document.page_content)["function"]
    except Exception as e:
        raise ValueError(f"Extraction failed: {e}")

    # Construct and return the GraphDocument
    return GraphDocument(
        nodes=[map_to_base_node(node) for node in extracted_data.nodes],
        relationships=[map_to_base_relationship(rel) for rel in extracted_data.rels],
        source=document,
    )
