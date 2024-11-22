from openai import OpenAI
import os
from utils import env_parse
from langchain_community.graphs import Neo4jGraph
from graph_builder import build_graph
from get_communities import get_communities, summarize_communities
from graph_nx import build_nx_graph
from typing import List
import networkx as nx
import pickle

with open("graphRAG/openai.txt", "r") as f:
    api_key = f.read()
credentials = env_parse("Neo4j-8e91ca29-Created-2024-11-21.txt")
url = credentials["NEO4J_URI"]
username = credentials["NEO4J_USERNAME"]
password = credentials["NEO4J_PASSWORD"]
url = url.replace("neo4j+s://", "neo4j+ssc://")
graph = Neo4jGraph(url=url, username=username, password=password)

client = OpenAI(api_key=api_key)


def main(file_paths: List[str], graph: Neo4jGraph, query: str) -> str:
    """
    Main function to extract the graph, get communities, summarize them, and generate a final answer.

    Args:
    file_paths: List of file paths to be processed
    graph: Neo4jGraph object
    query: Query to be answered

    Returns:
    nx.Graph: NetworkX graph
    List[str]: List of community summaries
    """
    graph_document = build_graph(file_paths, graph)
    G = build_nx_graph(graph_document)
    communities = get_communities(G)
    community_summaries = summarize_communities(communities, G, client)
    return G, community_summaries


if __name__ == "__main__":
    file_paths = ["test/doc{i}.txt".format(i=i) for i in range(1, 6)]
    query = "What factors in these articles can impact medical inflation in the UK in the short term? Answer in details with examples from the summaries."
    G, community_summaries = main(file_paths, graph, query)
    with open("graphRAG/graph.gpickle", "wb") as f:
        pickle.dump(G, f)
    # save community summaries to a file
    with open("graphRAG/community_summaries.pkl", "wb") as f:
        pickle.dump(community_summaries, f)

    # final_answer = generate_answer(community_summaries, query, client)
    # print("Query:", query)
    # print("Final answer:", final_answer)
