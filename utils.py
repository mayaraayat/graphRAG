from uuid import uuid4
from KG_classes import FileNode, ChunkNode, Property, Node, Relationship
from langchain_community.graphs.graph_document import (
    Node as BaseNode,
    Relationship as BaseRelationship,
)
from langchain.schema import Document
import os
from typing import List
from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader


def env_parse(file):
    credentials = {}
    with open(file, "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            line = line.split("=")
            credentials[line[0]] = line[1].strip()
    return credentials


def create_file_node(file_path: str) -> FileNode:
    """Create a file node."""
    return FileNode(
        id=file_path.split("/")[-1],
        type="File",
        properties=[
            Property(key="path", value=file_path),
            Property(key="name", value=os.path.basename(file_path)),
        ],
    )


def create_chunk_node(
    chunk: Document, chunk_idx: int, file_node: FileNode
) -> ChunkNode:
    """Create a chunk node."""
    return ChunkNode(
        id=file_node.id + str(chunk_idx),
        type="Chunk",
        properties=[
            Property(key="content", value=chunk.page_content),
            Property(key="idx", value=str(chunk_idx)),
            Property(key="sourceFileId", value=file_node.id),
        ],
    )


def format_property_key(s: str) -> str:
    words = s.split()
    if not words:
        return s
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)


def props_to_dict(props) -> dict:
    """Convert properties to a dictionary."""
    properties = {}
    if not props:
        return properties
    for p in props:
        properties[format_property_key(p.key)] = p.value
    return properties


def map_to_base_node(node: Node) -> BaseNode:
    """Map the KnowledgeGraph Node to the base Node."""
    if type(node.properties) == dict:
        properties = node.properties
    else:
        properties = props_to_dict(node.properties) if node.properties else {}
    # Add name property for better Cypher statement generation
    properties["name"] = node.id.title()
    return BaseNode(
        id=node.id.title(), type=node.type.capitalize(), properties=properties
    )


def map_to_base_relationship(rel: Relationship) -> BaseRelationship:
    """Map the KnowledgeGraph Relationship to the base Relationship."""
    source = map_to_base_node(rel.source)
    target = map_to_base_node(rel.target)
    properties = props_to_dict(rel.properties) if rel.properties else {}
    return BaseRelationship(
        source=source, target=target, type=rel.type.capitalize(), properties=properties
    )


def create_relationship(source: Node, target: Node, type: str):
    source = map_to_base_node(source)
    target = map_to_base_node(target)
    return BaseRelationship(
        source=source, target=target, type=type.capitalize(), properties={}
    )


def load_and_split_documents(
    file_paths: List[str], chunk_size: int = 100, chunk_overlap: int = 20
):
    """
    Load and split multiple documents into chunks.

    Args:
        file_paths (List[str]): List of file paths to load.
        chunk_size (int): Size of each chunk (in tokens).
        chunk_overlap (int): Overlap between chunks (in tokens).

    Returns:
        List: List of split document chunks.
    """
    all_pages = []
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    for file_path in file_paths:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)

        pages = loader.load_and_split()
        chunks = text_splitter.split_documents(pages)
        all_pages.extend(chunks)

    return all_pages
