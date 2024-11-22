# **GraphRAG**

GraphRAG is a Python-based framework designed to build, manage, and query **Graph Retrieval-Augmented Generation (GraphRAG)** systems. It combines advanced graph processing tools like **Neo4j** and **NetworkX** with natural language capabilities powered by **LLMs (Large Language Models)** to create, visualize, and query knowledge graphs.

---

## **Features**
- **Knowledge Graph Construction**: Extract entities and relationships from text and construct a knowledge graph.
- **Graph Management**:
  - Store and query graphs in **Neo4j**.
  - Perform analysis and operations on **NetworkX** graphs.
- **Graph Summarization**: Analyze and summarize communities within the graph.
- **Natural Language Interaction**: Generate answers to queries using the knowledge graph and language models.
- **Interactive Visualization**: Explore and interact with knowledge graphs using **Gradio**.
- **Modular Design**: A highly organized and reusable codebase for scalable development.

---

## **File Structure**

├── graphRAG # Main project directory

    ├── init.py #Marks the directory as a Python package 

    ├── main.py # Entry point to run the GraphRAG pipeline 

    ├── community_summaries.pkl # Serialized summaries of graph communities 

    ├── entities_extraction.py # Extract entities and relationships from text 

    ├── generating_answers.py # Generate answers to queries using the graph 

    ├── get_communities.py # Identify and analyze graph communities 

    ├── gradio_viz.py # Gradio-based interactive graph visualization 

    ├── graph_builder.py # Build the knowledge graph 

    ├── graph_nx.py # Manage and analyze NetworkX-based graphs 

    ├── KG_classes.py # Defines classes for knowledge graph objects 

    ├── subgraphs.py # Extract and manage subgraphs 

    ├── utils.py # Shared utility functions 

    ├── graph.gpickle # Serialized NetworkX graph file 

    ├── README.md # Project documentation (this file)