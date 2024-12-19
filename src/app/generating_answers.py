from openai import OpenAI
from src.app.utils.utils import read_prompt


def generate_answer(community_summaries: list[str], query: str, client: OpenAI) -> str:
    """
    Generate a final answer by combining answers from different community summaries.

    Args:
        community_summaries (list[str]): List of summaries for each community.
        query (str): Query to be answered.
        client (OpenAI): OpenAI client for generating answers.

    Returns:
        str: Final answer generated by combining the answers from different communities.
    """
    intermediate_answers = []
    for index, summary in enumerate(community_summaries):
        print(f"Answering community {index+1}/{len(community_summaries)}:")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": read_prompt(
                        "GraphRAG_vf/src/prompts/system_prompts/intermediate_answers.txt"
                    ),
                },
                {"role": "user", "content": f"Query: {query} Summary: {summary}"},
            ],
        )
        print("Intermediate answer:", response.choices[0].message.content)
        intermediate_answers.append(response.choices[0].message.content)

    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": read_prompt(
                    "GraphRAG_vf/src/prompts/system_prompts/final_answers.txt"
                ),
            },
            {
                "role": "user",
                "content": f"Intermediate answers: {intermediate_answers} , Query: {query}",
            },
        ],
    )
    final_answer = final_response.choices[0].message.content
    return final_answer