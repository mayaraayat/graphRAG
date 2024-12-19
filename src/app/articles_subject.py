from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util


def calculate_tfidf_similarity(articles: list[str], theme: str) -> list[float]:
    """
    Calculate cosine similarity between articles and a theme using TF-IDF.

    Args:
        articles (list[str]): List of articles as strings.
        theme (str): The theme to compare the articles against.

    Returns:
        list[float]: A list of cosine similarity scores between the theme and each article.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(articles + [theme])
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    return cosine_similarities.flatten()


def calculate_bert_similarity(
    articles: list[str], theme: str, model_name: str = "all-MiniLM-L6-v2"
) -> list[float]:
    """
    Calculate cosine similarity between articles and a theme using a BERT-based model.

    Args:
        articles (list[str]): List of articles as strings.
        theme (str): The theme to compare the articles against.
        model_name (str): Name of the SentenceTransformer model to use (default: "all-MiniLM-L6-v2").

    Returns:
        list[float]: A list of cosine similarity scores between the theme and each article.
    """
    # Load the SentenceTransformer model
    model = SentenceTransformer(model_name)

    # Encode articles and theme into embeddings
    article_embeddings = model.encode(articles, convert_to_tensor=True)
    theme_embedding = model.encode(theme, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_similarities = util.cos_sim(theme_embedding, article_embeddings)
    return cosine_similarities.flatten().cpu().numpy()
