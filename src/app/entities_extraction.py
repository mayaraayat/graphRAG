import os
from dotenv import load_dotenv
from langchain.chains.openai_functions import create_structured_output_chain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from src.KG_classes import KnowledgeGraph


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("Missing OPENAI_API_KEY. Please set it in the environment.")


if not api_key:
    raise EnvironmentError("Missing OPENAI_API_KEY. Please set it in the environment.")


def initialize_llm() -> ChatOpenAI:
    """
    Initialize the ChatOpenAI instance with required configuration.

    Returns:
        ChatOpenAI: Configured LLM instance.
    """
    return ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)


def get_extraction_chain(
    allowed_nodes: list[str] | None = None, allowed_rels: list[str] | None = None
):
    """
    Create a structured output chain for extracting knowledge graph data.

    Args:
        allowed_nodes (list[str] | None): List of allowed node labels, if any.
        allowed_rels (list[str] | None): List of allowed relationship types, if any.

    Returns:
        StructuredOutputChain: The chain for extracting knowledge graph data.
    """

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
    llm = initialize_llm()
    return create_structured_output_chain(KnowledgeGraph, llm, prompt, verbose=False)
