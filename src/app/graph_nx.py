import matplotlib.pyplot as plt
import networkx as nx
from langchain_community.graphs.graph_document import GraphDocument


def build_nx_graph(final_graph_document: GraphDocument) -> nx.Graph:
    """
    Build a NetworkX graph from a GraphDocument.

    Args:
        final_graph_document (GraphDocument): The graph document containing nodes and relationships.

    Returns:
        nx.Graph: A NetworkX graph with nodes and edges based on the input document.
    """
    # Extract unique node IDs
    nodes = {node.id for node in final_graph_document.nodes}
    relationships = final_graph_document.relationships

    # Initialize the graph and add nodes
    G = nx.Graph()
    G.add_nodes_from(nodes)

    # Add edges based on relationships
    for relation in relationships:
        G.add_edge(relation.source.id, relation.target.id, type=relation.type)

    return G


def plot_nx_graph(G: nx.Graph):
    """
    Plot a NetworkX graph with nodes, edges, and labels.

    Args:
        G (nx.Graph): The NetworkX graph to be visualized.

    Returns:
        None: Displays the graph visualization using Matplotlib.
    """
    pos = nx.spring_layout(G)  # Generate positions for nodes
    plt.figure(figsize=(20, 20))  # Configure the plot size

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=500, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color="black", width=5.0, alpha=0.8)

    # Draw labels for nodes
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

    # Draw labels for edges (showing the "type" attribute)
    edge_labels = nx.get_edge_attributes(G, "type")
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=5, font_color="red"
    )

    # Final plot adjustments
    plt.title("Knowledge Graph")
    plt.axis("off")  # Turn off axes for better visualization
    plt.show()
