from typing import List
from openai import OpenAI


def generate_answer(community_summaries: List[str], query: str, client: OpenAI) -> str:
    """
    Generate a final answer by combining the answers from different communities.

    Args:
    community_summaries: List of summaries for each community
    query: Query to be answered

    Returns:
    str: Final answer generated by combining the answers from different communities."""

    intermediate_answers = []
    for index, summary in enumerate(community_summaries):
        print(f"Answering community {index+1}/{len(community_summaries)}:")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Answer the following query based on the provided summary.",
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
                "content": "Combine these answers into a final response with important details and examples from these answers.",
            },
            {
                "role": "user",
                "content": f"Intermediate answers: {intermediate_answers}",
            },
        ],
    )
    final_answer = final_response.choices[0].message.content
    return final_answer
