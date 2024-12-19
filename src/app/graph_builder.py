from datetime import datetime
from tqdm import tqdm
from src.app.utils.utils import (
    create_file_node,
    create_chunk_node,
    create_relationship,
    load_and_split_documents,
    map_to_base_node,
)
from src.app.subgraphs import extract_graph
from langchain.schema import Document
from langchain_community.graphs.graph_document import GraphDocument


def build_graph(file_paths: list[str]) -> GraphDocument:
    """
    Build a graph from a list of file paths.

    Args:
        file_paths (list[str]): List of file paths to process.

    Returns:
        GraphDocument: A graph document containing nodes and relationships extracted from the files.
    """
    start_time = datetime.now()

    file_nodes = []
    chunk_nodes = []
    relationships = []
    distinct_nodes = []

    for file in file_paths:
        # Create a node for the file
        file_node = create_file_node(file)
        file_nodes.append(file_node)

        # Load and split the file into document chunks
        documents = load_and_split_documents([file])

        for idx, doc in tqdm(
            enumerate(documents), total=len(documents), desc=f"Processing {file}"
        ):
            # Create a node for each document chunk
            chunk_node = create_chunk_node(doc, idx, file_node)
            chunk_nodes.append(chunk_node)

            # Add a relationship from the chunk to the file
            relationships.append(create_relationship(chunk_node, file_node, "From"))

            # Extract a subgraph from the chunk
            graph_document = extract_graph(doc)

            # Add unique nodes and relationships from the subgraph
            for node in graph_document.nodes:
                if node.id not in {n.id for n in distinct_nodes}:
                    distinct_nodes.append(node)
                relationships.append(create_relationship(node, chunk_node, "From"))

            relationships.extend(graph_document.relationships)

    # Combine all nodes and relationships into the final graph document
    final_graph_document = GraphDocument(
        nodes=list(distinct_nodes)
        + [map_to_base_node(node) for node in file_nodes + chunk_nodes],
        relationships=relationships,
        source=Document(
            page_content="Combined source of all files and chunks",
            metadata={"description": "Generated from multiple files and their chunks"},
        ),
    )

    print(f"Graph built in {datetime.now() - start_time}")
    return final_graph_document
