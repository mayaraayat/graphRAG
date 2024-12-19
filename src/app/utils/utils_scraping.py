import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import pandas as pd


def fetch_article_content(url: str) -> tuple[str | None, str | None, str | None]:
    """
    Fetch the content of an article from a given URL.

    Args:
        url (str): The URL of the article to fetch.

    Returns:
        tuple: A tuple containing the title, date, and content of the article.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, "html.parser")

        title = soup.find("title").get_text() if soup.find("title") else None
        date = soup.find("time").get_text() if soup.find("time") else None
        article = soup.find("article")

        if article:
            content = article.get_text(strip=True)
            return title, date, content
    except requests.exceptions.Timeout:
        print(f"Request to {url} timed out.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch {url}: {e}")

    return None, None, None


def save_article_to_file(
    title: str,
    date: str,
    content: str,
    site_name: str,
    article_index: int,
    output_dir: str = "articles",
):
    """
    Save an article to a text file.

    Args:
        title (str): Title of the article.
        date (str): Publication date of the article.
        content (str): Content of the article.
        site_name (str): Name of the website.
        article_index (int): Index of the article.
        output_dir (str, optional): Directory to save the article file. Default is "articles".
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{site_name}_article_{article_index}.txt")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(f"Title: {title}\n")
        file.write(f"Date: {date}\n\n")
        file.write(content)


def process_article_urls(
    url_list: list[str], save_to_txt: bool = False, site_name: str | None = None
) -> pd.DataFrame:
    """
    Process a list of article URLs to fetch their content.

    Args:
        url_list (list[str]): List of article URLs.
        save_to_txt (bool, optional): Whether to save articles as text files. Default is False.
        site_name (str, optional): Name of the website. If None, it will be inferred from the URL.

    Returns:
        pd.DataFrame: A DataFrame containing the articles' URLs, titles, dates, and contents.
    """
    data = []
    for index, url in enumerate(url_list):
        print(f"Processing article {index + 1}/{len(url_list)}: {url}")
        parsed_url = urlparse(url)
        site_name = site_name or parsed_url.netloc.replace("www.", "").split(".")[0]

        title, date, content = fetch_article_content(url)
        if content:
            if save_to_txt:
                save_article_to_file(title, date, content, site_name, index)
            data.append({"url": url, "title": title, "date": date, "content": content})

    return pd.DataFrame(data)


def create_article_dataframe(
    url_list: list[str], save_to_txt: bool = False, site_name: str | None = None
) -> pd.DataFrame:
    """
    Create a DataFrame of articles by processing their URLs.

    Args:
        url_list (list[str]): List of article URLs.
        save_to_txt (bool, optional): Whether to save articles as text files. Default is False.
        site_name (str, optional): Name of the website. If None, it will be inferred from the URL.

    Returns:
        pd.DataFrame: A DataFrame containing the articles' URLs, titles, dates, and contents.
    """
    return process_article_urls(url_list, save_to_txt, site_name)


def save_dataframe_to_csv(
    df: pd.DataFrame, file_name: str, output_dir: str = "articles_df"
):
    """
    Save a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        file_name (str): Name of the CSV file.
        output_dir (str, optional): Directory to save the CSV file. Default is "articles_df".
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)
    df.to_csv(file_path, index=False)
    print(f"DataFrame saved to {file_path}")


def load_articles(directory: str, prefix: str, count: int) -> list[str]:
    """
    Load articles from text files.

    Args:
        directory (str): Directory containing article files.
        prefix (str): Prefix of the article files.
        count (int): Number of articles to load.

    Returns:
        list[str]: A list of article contents.
    """
    articles = []
    for i in range(count):
        file_path = os.path.join(directory, f"{prefix}_{i}.txt")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                articles.append(file.read())
    return articles


def save_articles_to_txt(df: pd.DataFrame, output_dir: str):
    """
    Save articles from a DataFrame to text files.

    Args:
        df (pd.DataFrame): DataFrame containing article data.
        output_dir (str): Directory to save the articles.
    """
    os.makedirs(output_dir, exist_ok=True)
    for _, row in df.iterrows():
        url = row.get("url", "Unknown URL")
        title = row.get("title", "Untitled").replace(" ", "_").replace("/", "_")
        date = row.get("date", "Unknown Date")
        content = row.get("content", "")

        filename = f"{title}.txt"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(f"URL: {url}\n")
            file.write(f"Title: {title}\n")
            file.write(f"Date: {date}\n\n")
            file.write(content)

    print(f"Articles saved to {output_dir}")
