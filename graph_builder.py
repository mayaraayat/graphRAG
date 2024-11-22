from datetime import datetime
from tqdm import tqdm
from typing import List
from utils import (
    create_file_node,
    create_chunk_node,
    create_relationship,
    load_and_split_documents,
)
from subgraphs import extract_graph
from langchain.schema import Document
from langchain_community.graphs.graph_document import GraphDocument
from utils import map_to_base_node
from langchain_community.graphs import Neo4jGraph


def build_graph(file_paths: List[str], graph: Neo4jGraph) -> GraphDocument:
    """
    Main function to build a graph from a list of file paths.

    Args:
    file_paths: List of file paths to be processed

    Returns:
    GraphDocument: Graph document containing the extracted graph

    """
    start_time = datetime.now()
    file_nodes = []
    chunk_nodes = []
    relationships = []
    distinct_nodes = []
    for file in file_paths:
        file_node = create_file_node(file)
        file_nodes.append(file_node)

        # Load and split documents
        documents = load_and_split_documents([file])

        for idx, doc in tqdm(enumerate(documents), total=len(documents)):
            chunk_node = create_chunk_node(doc, idx, file_node)
            chunk_nodes.append(chunk_node)
            relationships.append(create_relationship(chunk_node, file_node, "From"))

            graph_document = extract_graph(doc)
            # Get distinct nodes
            for node in graph_document.nodes:
                if node.id not in distinct_nodes:
                    distinct_nodes.append(node)
                relationships.append(create_relationship(node, chunk_node, "From"))
            # Get all relations
            for relation in graph_document.relationships:
                relationships.append(relation)

    print(f"Time taken: {datetime.now() - start_time}")

    final_graph_document = GraphDocument(
        nodes=distinct_nodes
        + [map_to_base_node(node) for node in [*file_nodes, *chunk_nodes]],
        relationships=relationships,
        source=Document(
            page_content="Combined source of all files and chunks",
            metadata={"description": "Generated from multiple files and their chunks"},
        ),
    )

    # Add the graph document to the graph
    # graph.add_graph_document(final_graph_document)
    return final_graph_document
