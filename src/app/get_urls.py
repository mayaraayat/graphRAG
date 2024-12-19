import os
import random
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from datetime import datetime
from serpapi import GoogleSearch

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise EnvironmentError("Missing GOOGLE_API_KEY. Please set it in the environment.")


def fetch_sitemap_content(sitemap_url: str) -> ET.Element:
    """
    Fetch and parse the content of an XML sitemap.

    Args:
        sitemap_url (str): URL of the XML sitemap.

    Returns:
        ET.Element: Parsed XML root element.
    """
    response = requests.get(sitemap_url)
    response.raise_for_status()
    return ET.fromstring(response.content)


def filter_urls_by_date(
    root: ET.Element,
    namespace: dict,
    target_year: int,
    target_month: int,
    date_extractor,
    url_filter: callable = lambda url: True,
    num_articles: int = 20,
) -> list[str]:
    """
    Filter URLs from a sitemap XML by year and month.

    Args:
        root (ET.Element): Parsed XML root element.
        namespace (dict): XML namespaces.
        target_year (int): Target year for filtering.
        target_month (int): Target month for filtering.
        date_extractor (callable): Function to extract the date from XML elements.
        url_filter (callable): Function to filter URLs.
        num_articles (int): Number of articles to return.

    Returns:
        list[str]: Filtered URLs.
    """
    filtered_urls = []

    for url in root.findall("ns:url", namespace):
        loc_element = url.find("ns:loc", namespace)
        if loc_element is not None and url_filter(loc_element.text):
            date = date_extractor(url, namespace)
            if date and date.year == target_year and date.month == target_month:
                filtered_urls.append(loc_element.text)

    return random.sample(filtered_urls, min(num_articles, len(filtered_urls)))


def get_filtered_urls_for_bbc(
    sitemap_url: str, target_year: int, target_month: int, num_articles: int = 20
) -> list[str]:
    """
    Retrieve filtered URLs from the BBC sitemap by date.

    Args:
        sitemap_url (str): URL of the BBC sitemap.
        target_year (int): Target year for filtering.
        target_month (int): Target month for filtering.
        num_articles (int): Number of articles to return.

    Returns:
        list[str]: Filtered URLs.
    """
    root = fetch_sitemap_content(sitemap_url)
    namespace = {
        "ns": "http://www.sitemaps.org/schemas/sitemap/0.9",
        "news": "http://www.google.com/schemas/sitemap-news/0.9",
    }

    def extract_date(url, ns):
        lastmod_element = url.find("ns:lastmod", ns)
        if lastmod_element is not None:
            try:
                return datetime.fromisoformat(
                    lastmod_element.text.replace("Z", "+00:00")
                )
            except ValueError:
                return None

    return filter_urls_by_date(
        root,
        namespace,
        target_year,
        target_month,
        extract_date,
        url_filter=lambda url: url.startswith("https://www.bbc.com/news/"),
        num_articles=num_articles,
    )


def get_filtered_urls_for_economist(
    sitemap_url: str, target_year: int, target_month: int, num_articles: int = 20
) -> list[str]:
    """
    Retrieve filtered URLs from The Economist sitemap by date embedded in URLs.

    Args:
        sitemap_url (str): URL of The Economist sitemap.
        target_year (int): Target year for filtering.
        target_month (int): Target month for filtering.
        num_articles (int): Number of articles to return.

    Returns:
        list[str]: Filtered URLs.
    """
    root = fetch_sitemap_content(sitemap_url)
    namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    def extract_date(url, _):
        loc_element = url.find("ns:loc", namespace)
        if loc_element is not None:
            try:
                parts = loc_element.text.split("/")
                return datetime(int(parts[4]), int(parts[5]), int(parts[6]))
            except (IndexError, ValueError):
                return None

    return filter_urls_by_date(
        root,
        namespace,
        target_year,
        target_month,
        extract_date,
        num_articles=num_articles,
    )


def get_filtered_urls_for_google_news(query: str, num_articles: int) -> list[str]:
    """
    Retrieve filtered URLs from Google News search results.

    Args:
        query (str): Search query for Google News.
        num_articles (int): Number of articles to return.

    Returns:
        list[str]: Filtered URLs.
    """
    params = {
        "engine": "google_news",
        "q": query,
        "gl": "us",
        "hl": "en",
        "api_key": google_api_key,
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    news_results = results.get("news_results", [])

    filtered_urls = []
    for result in news_results[:num_articles]:
        if "link" in result:
            filtered_urls.append(result["link"])
        elif "stories" in result:
            filtered_urls.extend(story["link"] for story in result["stories"])

    return filtered_urls


def get_filtered_urls_for_nhs(
    sitemap_url: str, target_year: int, target_month: int, num_articles: int | None = 20
):
    """
    Retrieves the last n articles from an XML sitemap for a given month and year.

    Args:
        sitemap_url (str): The URL of the XML sitemap.
        target_year (int): The target year to filter articles.
        target_month (int): The target month to filter articles.
        num_articles (int, optional): The number of articles to return. Default is 20.

    Returns:
        list: A list of URLs of the last n filtered articles.
    """
    # Download the XML sitemap content
    response = requests.get(sitemap_url)
    response.raise_for_status()  # Check if the request was successful

    # Parse the XML with namespace handling
    root = ET.fromstring(response.content.decode("utf-8"))
    namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    # Filter articles by the given month and year
    filtered_urls = []
    for url in root.findall("ns:url", namespace):
        loc_element = url.find("ns:loc", namespace)
        if loc_element is not None:
            # Extract the date from the URL
            url_parts = loc_element.text.split("/")
            if len(url_parts) > 4:
                url_year = int(url_parts[3])
                url_month = int(url_parts[4])

                # Filter by month and year
                if url_year == target_year and url_month == target_month:
                    filtered_urls.append(loc_element.text)

    # Return the last n articles
    return random.sample(filtered_urls, min(num_articles, len(filtered_urls)))


def get_filtered_urls(
    sitemap_url: str, target_year: int, target_month: int, num_articles: int = 20
) -> list[str]:
    """
    Retrieve filtered URLs based on the source site and date.

    Args:
        sitemap_url (str): URL of the sitemap.
        target_year (int): Target year for filtering.
        target_month (int): Target month for filtering.
        num_articles (int): Number of articles to return.

    Returns:
        list[str]: Filtered URLs.
    """
    handlers = {
        "https://www.bbc.com/": get_filtered_urls_for_bbc,
        "https://www.economist.com/": get_filtered_urls_for_economist,
    }

    for prefix, handler in handlers.items():
        if sitemap_url.startswith(prefix):
            return handler(sitemap_url, target_year, target_month, num_articles)

    return []  # Default empty list for unsupported sources
