import os
from dotenv import load_dotenv
from openai import OpenAI
from src.app.graph_builder import build_graph
from src.app.get_communities import get_communities, summarize_communities
from src.app.graph_nx import build_nx_graph
from src.app.generating_answers import generate_answer
from src.app.utils.utils import save_to_pickle

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("Missing OPENAI_API_KEY. Please set it in the environment.")
client = OpenAI(api_key=api_key)


def main(file_paths: list[str]) -> tuple:
    """
    Extract the graph, identify communities, and summarize them.

    Args:
        file_paths (list[str]): List of file paths to be processed.

    Returns:
        tuple: A tuple containing the NetworkX graph (nx.Graph) and a list of community summaries (list[str]).
    """
    # Build the graph document from file paths
    graph_document = build_graph(file_paths)

    # Convert the graph document into a NetworkX graph
    graph = build_nx_graph(graph_document)

    # Identify communities and summarize them
    communities = get_communities(graph)
    community_summaries = summarize_communities(communities, graph, client)

    return graph, community_summaries


if __name__ == "__main__":
    # Define input directory and output paths
    input_dir = "test"
    graph_pickle_path = "output/graph.gpickle"
    summaries_pickle_path = "output/community_summaries.pkl"

    # Prepare file paths
    file_paths = [os.path.join(input_dir, file) for file in os.listdir(input_dir)]

    # Run the main process
    graph, community_summaries = main(file_paths)

    # Save the graph and summaries to files
    save_to_pickle(graph, graph_pickle_path)
    save_to_pickle(community_summaries, summaries_pickle_path)

    # Optional: Uncomment the following lines to generate a final answer from summaries
    # query = "What factors in these articles can impact medical inflation in the UK in the short term? "
    # final_answer = generate_answer(community_summaries, query, client)
    # print("Query:", query)
    # print("Final answer:", final_answer)
