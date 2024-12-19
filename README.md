# GraphRAG

GraphRAG is a modular framework for extracting, summarizing, and querying knowledge from unstructured articles. It employs graph-based techniques for organizing data and OpenAI APIs for generating insights, making it ideal for applications like research and decision-making.

## Table of Contents

- Overview
- Folder Structure
- Setup and Installation
- How to Use
- Main Functions
- File Details
- Future Enhancements

### Overview

GraphRAG extracts entities and relationships from articles, builds knowledge graphs, and summarizes insights. 

Key features include:
- Article scraping with support for major sites like BBC, NHS, and The Economist.
- Knowledge graph generation using extracted entities and relationships.
- Community detection and summarization from graphs.
- Querying the graph for answers to user-defined questions.

### Folder Structure
```bash
GraphRAG_vf/
├── bin/
│   └── interface.py        # Gradio-based user interface
├── src/
│   ├── app/
│   │   ├── utils/          # Utility functions for article handling
│   │   │   ├── __init__.py
│   │   │   ├── functions.py            # Reusable helper functions for the interface
│   │   │   ├── utils_scraping.py       # Utility functions for scraping and processing articles
│   │   │   └── utils.py                # General utility functions
│   │   ├── articles_subject.py    # Handles similarity calculations for articles
│   │   ├── entities_extraction.py # Functions for extracting entities and relationships
│   │   ├── generating_answers.py  # Handles answering queries using OpenAI APIs
│   │   ├── get_communities.py     # Functions for community detection and summarization
│   │   ├── get_urls.py            # Retrieves and filters article URLs
│   │   ├── graph_builder.py       # Builds knowledge graphs from data
│   │   ├── graph_nx.py            # Handles NetworkX graph generation and plotting
│   │   ├── scraping_pipeline.py   # Handles article scraping and filtering
│   │   └── subgraphs.py           # Functions for subgraph analysis
│   ├── prompts/             # Contains prompt templates for OpenAI API interactions        
│   └── KG_classes.py        # Knowledge Graph data structure definitions         
├── __init__.py            # Marks the directory as a Python package
├── __main__.py            # Entry point for running the pipeline
├── README.md              # Project documentation (you are here)
└── requirements.txt       # Python dependencies
```
### Setup and Installation

#### Prerequisites
Python 3.10+

An OpenAI API key (add to .env file as OPENAI_API_KEY)

A Google API key (for Google News, add as GOOGLE_API_KEY)
#### Installation
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Create a .env file in the root directory and add your API keys:
```bash
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
```
### How to Use

1. Run the interface:
```bash
python bin/interface.py
```
2. Use the interface to scrape articles, build graphs, and query the graph for insights.

3. View the results in the interface.

The interface supports:

- Scraping articles
- Building graphs and summarizing communities
- Querying the graph for insights

### Main Functions
1. Article Scraping

Use scraping_pipeline.py to scrape articles, filter them, and calculate relevance based on a query.

Example: 
```python
from src.app.scraping_pipeline import scraping_pipeline

website = "bbc"
sitemap_url = "https://www.bbc.com/sitemaps/https-sitemap-com-news-1.xml"
num_articles = 30
query = "health"
target_year = 2024
target_month = 12

all_articles, top_articles = scraping_pipeline(
    website, sitemap_url, num_articles, query, target_year, target_month
)
```
2.  Graph Construction

Use graph_builder.py to construct knowledge graphs from the articles.

Example:
```python
from src.app.graph_builder import build_graph

file_paths = ["articles/article_1.txt", "articles/article_2.txt"]
graph_document = build_graph(file_paths)
```
3. Community Detection and Summarization

Use get_communities.py to detect communities in the graph and summarize them.

Example:
```python
from src.app.get_communities import get_communities, summarize_communitiesfrom get_communities import get_communities, summarize_communities

communities = get_communities(graph)
summaries = summarize_communities(communities, graph, client)
```
4. Querying the Graph
Use generating_answers.py to query the graph and generate answers.

Example:
```python
from src.app.generating_answers import generate_answer

query = "What factors impact healthcare inflation?"
final_answer = generate_answer(community_summaries, query, client)
```

### File Details

- Prompts Directory:
Contains structured prompts used by the OpenAI API to extract entities, generate answers, and summarize content.


### Future Enhancements

- Add Support for More Websites: Extend scraping to other news sources.
- Integrate Visualization: Add richer graph visualizations directly in the interface.
- Improve Answer Generation: Enhance the answer generation process for more accurate insights.