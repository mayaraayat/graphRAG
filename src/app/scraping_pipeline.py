import os
import numpy as np
from src.app.get_urls import (
    get_filtered_urls_for_bbc,
    get_filtered_urls_for_economist,
    get_filtered_urls_for_nhs,
    get_filtered_urls_for_google_news,
)
from src.app.utils.utils_scraping import (
    create_article_dataframe,
    save_dataframe_to_csv,
    save_articles_to_txt,
)
from src.app.articles_subject import calculate_bert_similarity


def scraping_pipeline(
    website: str,
    sitemap_url: str,
    num_articles: int,
    query: str,
    target_year: int,
    target_month: int,
    lim_articles: int = 200,
    output_dir: str = "articles/",
):
    """
    Scrapes articles from a specified website, filters them based on the provided criteria, and saves the results.

    Args:
        website (str): Name of the website to scrape articles from.
            Supported values: "bbc", "the economist", "nhs", "google news".
        sitemap_url (str): URL of the sitemap to extract article URLs from.
        num_articles (int): Number of articles to return after filtering.
        query (str): Query to compare against articles.
        target_year (int): Target year for filtering articles.
        target_month (int): Target month for filtering articles.
        lim_articles (int, optional): Maximum number of articles to retrieve. Default is 200.
        output_dir (str, optional): Directory to save article text files. Default is "articles/".

    Returns:
        tuple: Two DataFrames:
            - all_articles_df (DataFrame): Contains all scraped articles.
            - top_articles_df (DataFrame): Contains top N articles based on similarity (or None for Google News).
    """
    # Retrieve filtered URLs based on the website
    if website == "bbc":
        filtered_urls = get_filtered_urls_for_bbc(
            sitemap_url, target_year, target_month, lim_articles
        )
    elif website == "the economist":
        filtered_urls = get_filtered_urls_for_economist(
            sitemap_url, target_year, target_month, lim_articles
        )
    elif website == "nhs":
        filtered_urls = get_filtered_urls_for_nhs(
            sitemap_url, target_year, target_month, lim_articles
        )
    elif website == "google news":
        filtered_urls = get_filtered_urls_for_google_news(query, num_articles)
    else:
        raise ValueError(f"Invalid website name: {website}")

    if not filtered_urls:
        raise ValueError(f"No articles found for {website} with the given criteria.")

    # Store all articles in a DataFrame
    all_articles_df = create_article_dataframe(
        filtered_urls, save_to_txt=False, site_name=website
    )
    save_dataframe_to_csv(all_articles_df, f"{website}_all_articles.csv")

    # If the website is Google News, return only the scraped articles
    if website == "google news":
        save_articles_to_txt(all_articles_df, output_dir)
        return None, all_articles_df

    # Process articles to find the top N based on similarity to the query
    preprocessed_articles = all_articles_df["content"].tolist()
    similarity_scores = calculate_bert_similarity(preprocessed_articles, query)
    top_indices = np.argsort(similarity_scores)[-num_articles:][::-1]
    top_articles_df = all_articles_df.iloc[top_indices]
    top_similarity_scores = similarity_scores[top_indices]

    # Log the top articles with their similarity scores
    for idx, score in zip(top_indices, top_similarity_scores):
        print(f"Similarity Score: {score}")
        print(f"Title: {all_articles_df.iloc[idx]['title']}")
        print(f"Content: {all_articles_df.iloc[idx]['content']}\n")

    # Save results
    save_dataframe_to_csv(top_articles_df, f"{website}_selected_articles.csv")
    save_articles_to_txt(top_articles_df, output_dir)
    return all_articles_df, top_articles_df


def test_scraping_pipeline():
    """
    Run test cases for the scraping pipeline.
    """
    test_cases = [
        {
            "website": "bbc",
            "sitemap_url": "https://www.bbc.com/sitemaps/https-sitemap-com-news-1.xml",
            "query": "health",
            "target_year": 2024,
            "target_month": 12,
            "num_articles": 30,
        },
        {
            "website": "nhs",
            "sitemap_url": "https://www.england.nhs.uk/sitemap-posttype-post.2024.xml",
            "query": "health",
            "target_year": 2024,
            "target_month": 11,
            "num_articles": 30,
        },
        # Add more test cases as needed
    ]

    for case in test_cases:
        print(f"Running test case for {case['website']}...")
        try:
            all_articles, top_articles = scraping_pipeline(
                website=case["website"],
                sitemap_url=case["sitemap_url"],
                num_articles=case["num_articles"],
                query=case["query"],
                target_year=case["target_year"],
                target_month=case["target_month"],
            )
            print(f"Test case for {case['website']} completed successfully.")
        except Exception as e:
            print(f"Error in test case for {case['website']}: {e}")


if __name__ == "__main__":
    test_scraping_pipeline()
