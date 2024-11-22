import gradio as gr
import time
import re
import pickle
from generating_answers import generate_answer
from openai import OpenAI

with open("community_summaries.pkl", "rb") as f:
    community_summaries = pickle.load(f)

with open("openai.txt", "r") as f:
    api_key = f.read()
client = OpenAI(api_key=api_key)


def summarize_community(community_index: int) -> str:
    """Returns the summary of the selected community."""
    if 0 <= community_index < len(community_summaries):
        return community_summaries[community_index]
    return "Invalid community index."


def postprocess_answer(answer: str) -> str:
    """Formats the query response using specific rules."""
    # Replace newlines with <br> for line breaks
    formatted_answer = answer.replace("\n", "<br>")
    # Convert **bold** to <b>bold</b>
    formatted_answer = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", formatted_answer)
    return f"<div style='text-align: justify; font-size: 1.1em; line-height: 1.6;'>{formatted_answer}</div>"


def generate_answer_with_loading(query: str):
    """Handles query generation and updates progress status."""
    # First yield: Processing status with no output yet
    yield "Processing...", ""

    time.sleep(2)  # Simulate a delay for processing
    raw_response = generate_answer(community_summaries, query, client)
    # Second yield: Ready status with processed answer
    yield "Ready", postprocess_answer(raw_response)


# Tab 2: Answer Query
with gr.Blocks() as query_tab:
    gr.Markdown("### Answer a Query")
    examples = [
        "What factors in these articles can impact medical inflation in the UK in the short term? Answer in details with examples from the summaries.",
        "How does public health crises affect medical costs?",
        "What are the regulatory challenges driving healthcare inflation?",
    ]
    query_example = gr.Dropdown(
        label="Select a Query Example", choices=examples, interactive=True
    )
    query = gr.Textbox(lines=5, label="Query", placeholder="Type your query here...")
    query_button = gr.Button("Get Answer")
    progress_bar = gr.Markdown("### Status: Ready", visible=True)
    query_output = gr.HTML(label="Answer")

    def update_query(selected_example: str):
        """Updates the query box with the selected example."""
        return selected_example

    query_example.change(update_query, inputs=query_example, outputs=query)
    query_button.click(
        generate_answer_with_loading, inputs=query, outputs=[progress_bar, query_output]
    )

# Group tabs into a single interface
with gr.Blocks() as demo:
    gr.Markdown("# Community Tools")
    with gr.Tabs():
        with gr.Tab("Answer Query"):
            query_tab.render()

# Launch the app
demo.launch(share=True)
