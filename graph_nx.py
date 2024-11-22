import matplotlib.pyplot as plt
import networkx as nx
from langchain_community.graphs.graph_document import GraphDocument


def build_nx_graph(final_graph_document: GraphDocument) -> nx.Graph:
    """
    Build a NetworkX graph from a GraphDocument.

    Args:
        final_graph_document (GraphDocument): The final graph document.

    Returns:
        nx.Graph: The NetworkX graph.
    """
    nodes = [node.id for node in final_graph_document.nodes]
    unique_nodes = set(nodes)
    relationships = final_graph_document.relationships
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for relation in relationships:
        G.add_edge(relation.source.id, relation.target.id, type=relation.type)
    return G


def plot_nx_graph(G: nx.Graph):
    """
    Plot the NetworkX graph.

    Args:
        G (nx.Graph): The NetworkX graph.
    """
    pos = nx.spring_layout(G)
    plt.figure(figsize=(20, 20))
    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=500, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color="black", width=5.0, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

    # Add edge labels (for the `type` attribute)
    edge_labels = nx.get_edge_attributes(G, "type")
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=5, font_color="red"
    )

    # Show the plot
    plt.title("Knowledge Graph")
    plt.axis("off")  # Turn off the axes for better visualization
    plt.show()
