import os
import pickle
from langchain.schema import Document
from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from src.KG_classes import FileNode, ChunkNode, Property, Node, Relationship
from langchain_community.graphs.graph_document import (
    Node as BaseNode,
    Relationship as BaseRelationship,
)


def save_to_pickle(data, file_path: str):
    """
    Save data to a pickle file.

    Args:
        data: The data to save.
        file_path (str): Path to the pickle file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def read_prompt(prompt_file: str) -> str:
    """
    Read the prompt from a file.

    Args:
    prompt_file: File containing the prompt.

    Returns:
    str: The prompt read from the file.
    """
    # base_path = os.path.join(
    #     *os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:-2]
    # )
    # prompt_file = os.path.join(base_path, prompt_file)
    with open(prompt_file, "r") as file:
        prompt = file.read()
    return prompt


def create_file_node(file_path: str) -> FileNode:
    """Create a file node representing the file's metadata.

    Args:
        file_path (str): Path to the file.

    Returns:
        FileNode: File node representing the file.
    """
    return FileNode(
        id=os.path.basename(file_path),
        type="File",
        properties=[
            Property(key="path", value=file_path),
            Property(key="name", value=os.path.basename(file_path)),
        ],
    )


def create_chunk_node(
    chunk: Document, chunk_idx: int, file_node: FileNode
) -> ChunkNode:
    """Create a chunk node representing a text chunk.

    Args:
        chunk (Document): Text chunk.
        chunk_idx (int): Index of the chunk.
        file_node (FileNode): File node representing the source file.

    Returns:
        ChunkNode: Chunk node representing the text chunk."""
    return ChunkNode(
        id=f"{file_node.id}_{chunk_idx}",
        type="Chunk",
        properties=[
            Property(key="content", value=chunk.page_content),
            Property(key="idx", value=str(chunk_idx)),
            Property(key="sourceFileId", value=file_node.id),
        ],
    )


def format_property_key(key: str) -> str:
    """Format a property key into camelCase.

    Args:
        key (str): Property key.

    Returns:
        str: Formatted property key."""
    words = key.split()
    return words[0].lower() + "".join(word.capitalize() for word in words[1:])


def props_to_dict(properties: list[Property]) -> dict[str, str]:
    """Convert a list of properties to a dictionary.

    Args:
        properties (list[Property]): List of properties.

    Returns:
        dict[str, str]: Dictionary of properties."""
    return {format_property_key(p.key): p.value for p in properties}


def map_to_base_node(node: Node) -> BaseNode:
    """Map a custom Node to a base Node for the graph.

    Args:
        node (Node): Custom Node to map.

    Returns:
        BaseNode: Base Node for the graph."""
    properties = (
        props_to_dict(node.properties)
        if isinstance(node.properties, list)
        else node.properties or {}
    )
    properties["name"] = node.id.title()
    return BaseNode(
        id=node.id.title(), type=node.type.capitalize(), properties=properties
    )


def map_to_base_relationship(rel: Relationship) -> BaseRelationship:
    """Map a custom Relationship to a base Relationship for the graph.

    Args:
        rel (Relationship): Custom Relationship to map.

    Returns:
        BaseRelationship: Base Relationship for the graph."""
    return BaseRelationship(
        source=map_to_base_node(rel.source),
        target=map_to_base_node(rel.target),
        type=rel.type.capitalize(),
        properties=props_to_dict(rel.properties) if rel.properties else {},
    )


def create_relationship(
    source: Node, target: Node, relationship_type: str
) -> BaseRelationship:
    """Create a base Relationship between two nodes.

    Args:
        source (Node): Source node.
        target (Node): Target node.
        relationship_type (str): Type of the relationship.

    Returns:
        BaseRelationship: Base Relationship between the two nodes."""
    return BaseRelationship(
        source=map_to_base_node(source),
        target=map_to_base_node(target),
        type=relationship_type.capitalize(),
        properties={},
    )


def load_and_split_documents(
    file_paths: list[str], chunk_size: int = 100, chunk_overlap: int = 20
) -> list[Document]:
    """
    Load documents from file paths and split them into chunks.

    Args:
        file_paths (list[str]): List of file paths to load.
        chunk_size (int): Size of each chunk (in tokens).
        chunk_overlap (int): Overlap between chunks (in tokens).

    Returns:
        list[Document]: List of split document chunks.
    """
    all_chunks = []
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    for file_path in file_paths:
        try:
            loader = (
                PyPDFLoader(file_path)
                if file_path.endswith(".pdf")
                else TextLoader(file_path)
            )
            pages = loader.load_and_split()
            all_chunks.extend(text_splitter.split_documents(pages))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return all_chunks
