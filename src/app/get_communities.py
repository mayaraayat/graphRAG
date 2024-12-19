import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
from openai import OpenAI
from tqdm import tqdm
from src.app.utils.utils import read_prompt


def get_partition(graph: nx.Graph) -> dict[str, int]:
    """
    Compute the partition of the graph using the Louvain method.

    Args:
        graph (nx.Graph): The NetworkX graph.

    Returns:
        dict[str, int]: A dictionary mapping nodes to their community IDs.
    """
    return community_louvain.best_partition(graph)


def plot_graph_with_communities(graph: nx.Graph, partition: dict[str, int]):
    """
    Plot the NetworkX graph with communities visualized by color.

    Args:
        graph (nx.Graph): The NetworkX graph.
        partition (dict[str, int]): A dictionary mapping nodes to their community IDs.
    """
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(graph)  # Use spring layout for visualization

    # Draw nodes with colors based on communities
    node_colors = [partition[node] for node in graph.nodes()]
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=500,
        cmap=plt.get_cmap("tab20"),
        node_color=node_colors,
        alpha=0.9,
    )
    nx.draw_networkx_edges(graph, pos, edge_color="black", width=5.0, alpha=0.8)
    nx.draw_networkx_labels(graph, pos, font_size=8, font_color="black")

    # Draw edge labels for relationships
    edge_labels = nx.get_edge_attributes(graph, "type")
    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels=edge_labels, font_size=5, font_color="red"
    )

    plt.title("Knowledge Graph with Communities")
    plt.axis("off")  # Hide axes for a cleaner visualization
    plt.show()


def get_communities(graph: nx.Graph) -> list[list[str]]:
    """
    Identify and group nodes into communities based on the graph structure.

    Args:
        graph (nx.Graph): The NetworkX graph.

    Returns:
        list[list[str]]: A list of communities, where each community is a list of node IDs.
    """
    partition = get_partition(graph)
    plot_graph_with_communities(graph, partition)
    num_communities = len(set(partition.values()))
    return [
        [node for node, community_id in partition.items() if community_id == i]
        for i in range(num_communities)
    ]


def summarize_communities(
    communities: list[list[str]], graph: nx.Graph, client: OpenAI
) -> list[str]:
    """
    Generate summaries for each community based on its entities and relationships.

    Args:
        communities (list[list[str]]): A list of communities.
        graph (nx.Graph): The NetworkX graph.
        client (OpenAI): The OpenAI client for generating summaries.

    Returns:
        list[str]: A list of summaries for each community.
    """
    community_summaries = []

    for index, community in tqdm(enumerate(communities), total=len(communities)):
        # Extract subgraph for the current community
        subgraph = graph.subgraph(community)
        nodes = list(subgraph.nodes)
        edges = list(subgraph.edges(data=True))

        # Describe nodes and relationships
        description = "Entities: " + ", ".join(nodes) + "\nRelationships: "
        relationships = [
            f"{source} -> {data.get('type', 'unknown')} -> {target}"
            for source, target, data in edges
        ]
        description += ", ".join(relationships)

        # Generate summary using OpenAI
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": read_prompt(
                            "GraphRAG_vf/src/prompts/system_prompts/community_summaries.txt"
                        ),
                    },
                    {"role": "user", "content": description},
                ],
            )
            summary = response.choices[0].message.content.strip()
            community_summaries.append(summary)
        except Exception as e:
            print(f"Failed to summarize community {index}: {e}")
            community_summaries.append(f"Error summarizing community {index}")

    return community_summaries
