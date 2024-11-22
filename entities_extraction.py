import os
from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import List, Optional
from KG_classes import KnowledgeGraph

with open("graphRAG/openai.txt", "r") as f:
    api_key = f.read()
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)


def get_extraction_chain(
    allowed_nodes: Optional[List[str]] = None, allowed_rels: Optional[List[str]] = None
):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""# Knowledge Graph Extraction Instructions
    ## 1. Purpose
    You are a state-of-the-art system for extracting structured data to construct a **knowledge graph**. The graph consists of:
    - **Nodes**: Entities or concepts.
    - **Relationships**: Connections between nodes representing their interactions or associations.

    ## 2. Guidelines for Nodes
    - Use **general labels** for node types (e.g., "person", "organization").
    - Use **human-readable identifiers** for node IDs (no integers or generic IDs).
    - Include attributes as key-value pairs with `camelCase` keys (e.g., `birthDate: "1990-01-01"`).
    - Do **not create separate nodes** for numerical data or dates; these should always be node properties.

    {'- **Allowed Node Labels:** ' + ", ".join(allowed_nodes) if allowed_nodes else ""}
    ## 3. Guidelines for Relationships
    - Clearly define relationships between nodes, using concise and meaningful labels (e.g., "worksAt", "bornIn").
    - Only use relationships allowed in the context.
    {'- **Allowed Relationship Types:** ' + ", ".join(allowed_rels) if allowed_rels else ""}
    - Avoid overly detailed or complex relationship labels.

    ## 4. Coreference Resolution
    - Use the most complete identifier for entities across the graph. For example, use "John Doe" instead of "John" or "he".

    ## 5. Strict Compliance
    Adhere to these rules exactly to ensure the generated graph is clear, coherent, and consistent.""",
            ),
            (
                "human",
                "Extract information from the following text using these rules: {input}",
            ),
            ("human", "Ensure the output is in the correct format."),
        ]
    )
    return create_structured_output_chain(KnowledgeGraph, llm, prompt, verbose=False)
