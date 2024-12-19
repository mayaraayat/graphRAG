import os
import re
import pickle
import gradio as gr
import networkx as nx
from openai import OpenAI
from dotenv import load_dotenv

from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.graphs import Neo4jGraph
from src.app.graph_builder import build_graph
from src.app.graph_nx import build_nx_graph
from src.app.get_communities import get_communities, summarize_communities
from src.app.generating_answers import generate_answer
from src.app.utils.utils_scraping import save_articles_to_txt, process_article_urls
from src.app.scraping_pipeline import scraping_pipeline


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("Missing OPENAI_API_KEY. Please set it in the environment.")

client = OpenAI(api_key=api_key)


# File Management Functions
def handle_source_selection(source: str, data_folder: str) -> str:
    """
    Display the content of the selected source file.

    Args:
        source (str): Name of the source file to display.
        data_folder (str): Path to the folder containing the source files.

    Returns:
        str: Content of the selected source file, or an error message if no source is selected.
    """
    if not source:
        return "No source selected."
    return load_file_content(source, data_folder)


def load_file_content(filename: str, data_folder: str) -> str:
    """
    Load the content of a specific file from the data folder.

    Args:
        filename (str): Name of the file to load.
        data_folder (str): Path to the folder containing the file.

    Returns:
        str: Content of the file, or an error message if the file is not found.
    """
    file_path = os.path.join(data_folder, filename)
    if not os.path.exists(file_path):
        return "File not found."
    with open(file_path, "r") as file:
        return file.read()


def read_article_titles(data_folder: str = "articles") -> list[str]:
    """
    Retrieve article titles from a specified folder.

    Args:
        data_folder (str): Path to the folder containing articles.

    Returns:
        list[str]: List of article filenames ending in '.txt'.
    """
    if not os.path.exists(data_folder):
        return []
    return [file for file in os.listdir(data_folder) if file.endswith(".txt")]


def remove_article(title: str, articles_dir: str = "articles") -> list[str]:
    """
    Remove an article by its title and return updated titles.

    Args:
        title (str): Title of the article to remove.
        articles_dir (str): Path to the articles directory.

    Returns:
        list[str]: Updated list of article titles.
    """
    file_path = os.path.join(articles_dir, title)
    if os.path.exists(file_path):
        os.remove(file_path)
    return read_article_titles(data_folder=articles_dir)


def add_article(url: str, output_dir: str = "articles") -> list[str]:
    """
    Add an article from a URL and save it to the specified directory.

    Args:
        url (str): URL of the article to add.
        output_dir (str): Directory to save the article.

    Returns:
        list[str]: Updated list of article titles.
    """
    article_df = process_article_urls([url])
    save_articles_to_txt(article_df, output_dir)
    return read_article_titles(data_folder=output_dir)


def update_sitemap_url(website: str) -> gr.Textbox:
    """
    Get the sitemap URL for a specified website.

    Args:
        website (str): Website name (e.g., 'bbc', 'the economist').

    Returns:
        gr.Textbox: Updated sitemap URL.
    """
    sitemap_urls = {
        "bbc": "https://www.bbc.com/sitemaps/https-sitemap-com-news-1.xml",
        "the economist": "https://www.economist.com/sitemap-2024-Q4.xml",
        "nhs": "https://www.england.nhs.uk/sitemap-posttype-post.2024.xml",
        "google news": "",
    }
    return gr.update(value=sitemap_urls.get(website, ""))


def scraping_interface(
    website: str,
    sitemap_url: str,
    num_articles: int,
    query: str,
    target_year: int,
    target_month: int,
    data_folder: str,
) -> tuple[str, list[list[str]]]:
    """
    Execute the scraping pipeline for a specified website and update article titles.

    Args:
        website (str): Website to scrape articles from.
        sitemap_url (str): Sitemap URL for the website.
        num_articles (int): Maximum number of articles to scrape.
        query (str): Search query for filtering articles.
        target_year (int): Target year for the articles.
        target_month (int): Target month for the articles.
        data_folder (str): Directory to save the scraped articles.

    Returns:
        tuple[str, list[list[str]]]: Scraping status message and updated list of article titles in a tabular format.
    """
    # Run the scraping pipeline
    _, top_articles_df = scraping_pipeline(
        website=website,
        sitemap_url=sitemap_url,
        num_articles=num_articles,
        query=query,
        target_year=target_year,
        target_month=target_month,
        output_dir=data_folder,
    )

    # Retrieve updated article titles
    titles = [[title] for title in read_article_titles(data_folder)]

    # Return completion message and updated titles
    return f"Scraping completed. {len(top_articles_df)} articles saved.", titles


# Data Processing Functions
def process_data_folder(data_folder: str) -> list[str]:
    """
    Process a data folder and return a list of file paths.

    Args:
        data_folder (str): Path to the folder containing data files.

    Returns:
        list[str]: List of file paths or error string.
    """
    if not os.path.exists(data_folder):
        return f"Error: The folder '{data_folder}' does not exist."
    return [
        os.path.join(data_folder, file)
        for file in os.listdir(data_folder)
        if os.path.isfile(os.path.join(data_folder, file))
    ]


def build_graph_and_summarize(
    data_folder: str, graph_pickle: str, summary_pickle: str
) -> tuple[nx.Graph, list[str]]:
    """
    Build a graph and summarize communities from files in a data folder.

    Args:
        data_folder (str): Path to the data folder.
        graph_pickle (str): Path to save the graph pickle file.
        summary_pickle (str): Path to save the community summaries.

    Returns:
        tuple[nx.Graph, list[str]]: NetworkX graph and list of community summaries.
    """
    try:
        file_paths = process_data_folder(data_folder)
        if isinstance(file_paths, str):  # Error message
            return file_paths, None
        graph_document = build_graph(file_paths)
        G = build_nx_graph(graph_document)
        communities = get_communities(G)
        community_summaries = summarize_communities(communities, G, client)

        # Save the graph and summaries
        with open(graph_pickle, "wb") as f:
            pickle.dump(G, f)
        with open(summary_pickle, "wb") as f:
            pickle.dump(community_summaries, f)

        return G, community_summaries
    except Exception as e:
        return f"Error: {str(e)}", None


def display_graph_summary(data_folder: str, session_id: str) -> str:
    """
    Display a summary of the graph and its communities.

    Args:
        data_folder (str): Path to the data folder.
        session_id (str): Unique session ID.

    Returns:
        str: Summary string or error message.
    """
    G, community_summaries = build_graph_and_summarize(
        data_folder, f"{session_id}.gpickle", f"{session_id}.pkl"
    )
    if isinstance(G, str):  # Error occurred
        return G
    return "\n".join(
        [
            f"Community {i + 1}: {summary}"
            for i, summary in enumerate(community_summaries)
        ]
    )


def extract_sources_and_load_content(answer: str, data_folder: str) -> list[str]:
    """
    Extract unique source filenames from the answer text and perform partial matching
    against files in the specified data folder.

    Args:
        answer (str): Text containing source references.
        data_folder (str): Path to the folder containing source files.

    Returns:
        list[str]: List of unique matched source filenames.
    """
    # Extract source names using regex
    pattern = r"\(Source: ([^)]*?)\)"
    matches = re.findall(pattern, answer)

    # Normalize and clean source names
    sources = {
        src.strip().strip('"_').split(".")[0].lower()
        for match in matches
        for src in match.split(",")
    }

    # Collect original source filenames in the data folder
    original_sources = {
        file.lower(): file
        for file in os.listdir(data_folder)
        if os.path.isfile(os.path.join(data_folder, file))
    }

    # Match extracted sources to original filenames
    matched_sources = {
        original_sources[original]
        for extracted in sources
        for original in original_sources
        if extracted in original
    }

    # Return the unique matches as a list
    return list(matched_sources)


def edit_response(answer: str, client: OpenAI) -> str:
    """
    Format the response into ranked bullet points using GPT-4.

    Args:
        answer (str): The unformatted response text.
        client (OpenAI): OpenAI client for making GPT-4 API calls.

    Returns:
        str: Formatted and ranked response as bullet points, or an error message if the API call fails.
    """
    # Define the prompt for GPT-4
    prompt = f"""
    You are a meticulous editor and ranking expert. Format the following response into clear, concise bullet points.
    Rank the points by relevance to private health insurance in the UK, starting with the most relevant.
    If any sources are general or unspecified, mark them with 'No source explicitly found'.
    
    Prioritize:
    - Insights that address costs, premium adjustments, claims, or operational efficiency in private health insurance.
    - Specific statistics or actionable insights.
    - Clearly cited sources over vague or general insights.

    Do not add titles or sections, just list the ranked bullet points. The response is:

    {answer}

    Output only the ranked bullet points in order of relevance.
    """

    # Execute the GPT-4 API call
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error during GPT-4 formatting: {str(e)}"


def answer_with_sources(
    query: str, community_summaries: list[str], data_folder: str
) -> tuple[str, list[str]]:
    """
    Generate an answer for a query and retrieve associated sources.

    Args:
        query (str): User query.
        community_summaries (list[str]): Summarized community data.
        data_folder (str): Path to the data folder.

    Returns:
        tuple[str, list[str]]: Answer text and list of source filenames.
    """
    if query == "Select an option...":
        return "Please select a valid query.", []
    response = generate_answer(community_summaries, query, client)
    response = edit_response(response, client)
    sources = extract_sources_and_load_content(response, data_folder)
    return response, sources


def handle_query(
    selected_query: str, user_query_input: str, data_folder: str, session_id: str
) -> tuple[str, gr.Dropdown]:
    """
    Handle query response and provide source filenames.

    Args:
        selected_query (str): Predefined query or 'Write a custom query'.
        user_query_input (str): User-defined query text.
        data_folder (str): Path to the data folder.
        session_id (str): Unique session ID.

    Returns:
        tuple[str, gr.Dropdown]: Query response and source dropdown update.
    """
    query = (
        user_query_input if selected_query == "Write a custom query" else selected_query
    )
    summary_pickle = f"{session_id}.pkl"
    with open(summary_pickle, "rb") as f:
        community_summaries = pickle.load(f)
    response, sources = answer_with_sources(query, community_summaries, data_folder)
    dropdown_update = gr.Dropdown(choices=sources, visible=bool(sources))
    return response, dropdown_update


def update_article_titles_df(data_folder: str) -> gr.update:
    """
    Update the article titles in a dataframe format.

    Args:
        data_folder (str): Path to the folder containing article files.

    Returns:
        gr.update: Updated dataframe with article titles.
    """
    titles = read_article_titles(data_folder)
    return gr.update(value=[[title] for title in titles])


def update_article_titles(data_folder: str) -> gr.update:
    """
    Update the dropdown choices with available article titles.

    Args:
        data_folder (str): Path to the folder containing article files.

    Returns:
        gr.update: Dropdown update with the list of article titles.
    """
    return gr.update(choices=read_article_titles(data_folder))


def handle_remove_article(title: str, data_folder: str) -> tuple[gr.update, gr.update]:
    """
    Handle the removal of an article and update both the dataframe and dropdown.

    Args:
        title (str): Title of the article to remove.
        data_folder (str): Path to the folder containing article files.

    Returns:
        tuple[gr.update, gr.update]: Updated dataframe and dropdown with remaining titles.
    """
    updated_titles = remove_article(title, data_folder)
    return (
        gr.update(value=[[t] for t in updated_titles]),
        gr.update(choices=updated_titles),
    )


def handle_add_article(url: str, data_folder: str) -> tuple[gr.update, gr.update]:
    """
    Handle adding an article from a URL and update both the dataframe and dropdown.

    Args:
        url (str): URL of the article to add.
        data_folder (str): Path to the folder where the article will be saved.

    Returns:
        tuple[gr.update, gr.update]: Updated dataframe and dropdown with new titles.
    """
    updated_titles = add_article(url, data_folder)
    return (
        gr.update(value=[[t] for t in updated_titles]),
        gr.update(choices=updated_titles),
    )


def toggle_textbox(selected_query: str) -> tuple[gr.update, str]:
    """
    Toggle the visibility of a textbox based on the selected query.

    Args:
        selected_query (str): The query selected by the user.

    Returns:
        tuple[gr.update, str]: Update object to control textbox visibility
        and the selected query text.
    """
    is_visible = selected_query == "Write a custom query"
    return gr.update(visible=is_visible), selected_query
