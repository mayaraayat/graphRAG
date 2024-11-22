from langchain_community.graphs.graph_document import (
    Node as BaseNode,
    Relationship as BaseRelationship,
)

from typing import List
from pydantic import Field, BaseModel


class Property(BaseModel):
    """A single property consisting of key and value."""

    key: str = Field(..., description="Key of the property.")
    value: str = Field(..., description="Value of the property.")


class Node(BaseNode):
    """Represents a graph node with properties."""

    properties: List[Property] = Field(
        default_factory=list, description="List of node properties."
    )


class Relationship(BaseRelationship):
    """Represents a graph relationship with properties."""

    properties: List[Property] = Field(
        default_factory=list, description="List of relationship properties."
    )


class KnowledgeGraph(BaseModel):
    """Generate a knowledge graph with entities and relationships."""

    nodes: List[Node] = Field(..., description="List of nodes in the knowledge graph.")
    rels: List[Relationship] = Field(
        ..., description="List of relationships in the knowledge graph."
    )


class FileNode(Node):
    pass


class ChunkNode(Node):
    pass
