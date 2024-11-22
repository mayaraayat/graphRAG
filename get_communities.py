import networkx as nx
from typing import List, Dict
import community as community_louvain
import matplotlib.pyplot as plt
from openai import OpenAI
from tqdm import tqdm


def get_partition(graph: nx.Graph) -> List[int]:
    """Get the partition of the graph.

    Args:
        graph (nx.Graph): The NetworkX graph.

    Returns:
        List[int]: The partition of the graph.
    """
    partition = community_louvain.best_partition(graph)
    return partition


def plot_graph_with_communities(G: nx.Graph, partition: Dict[str, int]):
    """
    Plot the NetworkX graph with communities.

    Args:
        G (nx.Graph): The NetworkX graph.
        partition (Dict[str, int]): The partition of the graph.
    """
    # Draw graph with communities

    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G)  # Use spring layout for better visualization

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=500, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color="black", width=5.0, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

    # Add edge labels (for the `type` attribute)
    edge_labels = nx.get_edge_attributes(G, "type")
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=5, font_color="red"
    )

    # Draw communities
    values = [partition.get(node) for node in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos, node_size=500, cmap=plt.get_cmap("tab20"), node_color=values
    )

    # Show the plot
    plt.title("Knowledge Graph with Communities")
    plt.axis("off")  # Turn off the axes for better visualization
    plt.show()


def get_communities(graph: nx.Graph) -> List[List[str]]:
    """Get the communities in the graph.

    Args:
        graph (nx.Graph): The NetworkX graph.

    Returns:
        List[List[str]]: The list of communities.
    """
    partition = get_partition(graph)
    plot_graph_with_communities(graph, partition)
    c = len(set(partition.values()))
    communities = [[k for k, v in partition.items() if v == j] for j in range(c)]
    return communities


def summarize_communities(
    communities: List[List[str]], graph: nx.Graph, client: OpenAI
) -> List[str]:
    """
    Summarize the communities of entities and relationships.

    Args:
        communities (List[List[str]]): The list of communities.
        graph (nx.Graph): The NetworkX graph.
        client (OpenAI): The OpenAI client.

    Returns:
        List[str]: The list of community summaries.
    """
    community_summaries = []
    for index, community in tqdm(enumerate(communities), total=len(communities)):
        subgraph = graph.subgraph(community)
        nodes = list(subgraph.nodes)
        edges = list(subgraph.edges(data=True))
        description = "Entities: " + ", ".join(nodes) + "\nRelationships: "
        relationships = []
        for edge in edges:
            source, target, data = edge
            relation = data.get("type", "")
            relationships.append(f"{source} -> {data['type']} -> {target}")
        description += ", ".join(relationships)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """Summarize the following community of entities and relationships. 
                    """,
                },
                {"role": "user", "content": description},
            ],
        )
        summary = response.choices[0].message.content.strip()
        community_summaries.append(summary)
    return community_summaries
