from pydantic import Field, BaseModel
from langchain_community.graphs.graph_document import Node as BaseNode, Relationship as BaseRelationship


class Property(BaseModel):
    """
    Represents a key-value pair property for a graph element.

    Attributes:
        key (str): Key of the property.
        value (str): Value of the property.
    """
    key: str = Field(..., description="Key of the property.")
    value: str = Field(..., description="Value of the property.")


class Node(BaseNode):
    """
    Represents a graph node with properties.

    Attributes:
        properties (list[Property]): List of properties associated with the node.
    """
    properties: list[Property] = Field(default_factory=list, description="Properties of the node.")


class Relationship(BaseRelationship):
    """
    Represents a graph relationship with properties.

    Attributes:
        properties (list[Property]): List of properties associated with the relationship.
    """
    properties: list[Property] = Field(default_factory=list, description="Properties of the relationship.")


class KnowledgeGraph(BaseModel):
    """
    Represents a knowledge graph containing nodes and relationships.

    Attributes:
        nodes (list[Node]): Nodes in the knowledge graph.
        rels (list[Relationship]): Relationships between nodes in the knowledge graph.
    """
    nodes: list[Node] = Field(..., description="Nodes in the knowledge graph.")
    rels: list[Relationship] = Field(..., description="Relationships in the knowledge graph.")


class FileNode(Node):
    """
    Represents a node specific to files.

    Inherits:
        Node: All properties and behaviors of a generic graph node.
    """
    pass


class ChunkNode(Node):
    """
    Represents a node specific to chunks.

    Inherits:
        Node: All properties and behaviors of a generic graph node.
    """
    pass
