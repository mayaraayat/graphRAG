import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
print(sys.path)

from src.app.utils.functions import (
    update_article_titles_df,
    update_article_titles,
    handle_remove_article,
    handle_add_article,
    update_sitemap_url,
    scraping_interface,
    display_graph_summary,
    handle_query,
    handle_source_selection,
    read_article_titles,
    toggle_textbox,
)
import gradio as gr

with gr.Blocks() as interface:
    with gr.Tabs():
        # Scraping Tab
        with gr.Tab("Scraping"):
            with gr.Blocks() as iface:
                with gr.Row():
                    with gr.Column():
                        # Input elements for scraping
                        website = gr.Dropdown(
                            choices=["bbc", "the economist", "nhs", "google news"],
                            label="Website",
                            value="bbc",
                        )
                        sitemap_url = gr.Textbox(
                            label="Sitemap URL (not required for Google News)",
                            value="https://www.bbc.com/sitemaps/https-sitemap-com-news-1.xml",
                        )
                        max_articles = gr.Number(
                            label="Maximum Number of Articles", value=30
                        )
                        query = gr.Textbox(label="Query", value="health")
                        target_year = gr.Number(label="Target Year", value=2024)
                        target_month = gr.Number(label="Target Month", value=12)
                        data_folder = gr.Textbox(
                            label="Data Folder Path",
                            placeholder="Enter the path to the folder containing data files (e.g., 'april2024')",
                            lines=1,
                        )
                        scraping_button = gr.Button("Start Scraping")
                        scraping_result = gr.Textbox(label="Scraping Result")

                    with gr.Column():
                        # Display and manage articles
                        article_titles = gr.Dataframe(
                            headers=["Article Titles"],
                            value=[],
                            label="Article Titles",
                        )
                        article_to_remove = gr.Dropdown(
                            choices=[],
                            label="Select Article to Remove",
                        )
                        remove_button = gr.Button("Remove Selected Article")
                        url_input = gr.Textbox(label="Enter URL to Add Article")
                        add_button = gr.Button("Add Article")

                        # Update article titles
                        article_titles.change(
                            fn=update_article_titles_df,
                            inputs=[data_folder],
                            outputs=article_titles,
                        )
                        article_to_remove.change(
                            fn=update_article_titles,
                            inputs=[data_folder],
                            outputs=article_to_remove,
                        )

                        # Add and remove articles
                        add_button.click(
                            fn=handle_add_article,
                            inputs=[url_input, data_folder],
                            outputs=[article_titles, article_to_remove],
                        )
                        remove_button.click(
                            fn=handle_remove_article,
                            inputs=[article_to_remove, data_folder],
                            outputs=[article_titles, article_to_remove],
                        )

                # Update sitemap URL based on website selection
                website.change(
                    fn=update_sitemap_url,
                    inputs=website,
                    outputs=sitemap_url,
                )

                # Start scraping process
                scraping_button.click(
                    fn=scraping_interface,
                    inputs=[
                        website,
                        sitemap_url,
                        max_articles,
                        query,
                        target_year,
                        target_month,
                        data_folder,
                    ],
                    outputs=[scraping_result, article_titles],
                ).then(
                    fn=lambda folder: gr.update(choices=read_article_titles(folder)),
                    inputs=[data_folder],
                    outputs=article_to_remove,
                )

        # Indexing Tab
        with gr.Tab("Indexing"):
            gr.Markdown("# Graph and Community Summarization Interface")
            gr.Markdown(
                "Input the folder containing data files, names for the pickle files, and click the button to process the graph and summarize communities."
            )

            data_folder_input = gr.Textbox(
                label="Data Folder Path",
                placeholder="Enter the path to the folder containing data files (e.g., './april2024')",
                lines=1,
            )
            session_id = gr.Textbox(
                label="Session ID",
                placeholder="Enter the session ID",
                lines=1,
            )
            graph_summary_output = gr.Textbox(
                label="Graph and Community Summaries",
                placeholder="Summaries will appear here after processing.",
                lines=10,
                interactive=False,
            )
            process_button = gr.Button("Index Graph and Summarize Communities")

            process_button.click(
                fn=display_graph_summary,
                inputs=[data_folder_input, session_id],
                outputs=[graph_summary_output],
            )

        # Querying Tab
        with gr.Tab("Querying"):
            gr.Markdown("# GraphRAG Query Interface")
            gr.Markdown(
                "Input your query or select one of the suggested questions. "
                "The system will analyze the documents and return a detailed response."
            )

            data_folder_input_vis = gr.Textbox(
                label="Data Folder Path",
                placeholder="Enter the path to the folder containing data files (e.g., './april2024')",
                lines=1,
            )
            session_id = gr.Textbox(
                label="Session ID",
                placeholder="Enter the session ID",
                lines=1,
            )
            query_choices = gr.Dropdown(
                choices=[
                    "Select an option...",
                    "Write a custom query",
                    "Identify factors that can impact medical inflation in the UK",
                    "What are the long-term drivers of healthcare inflation?",
                    "How does public health crises affect medical costs?",
                    "What are the regulatory challenges driving healthcare inflation?",
                    "What are the events that could lead to a rise in healthcare services demand?",
                    "How do demographic changes affect medical costs?",
                    "Has there been any advancement in medical technology that could significantly increase treatment costs?",
                    "How do public health initiatives impact healthcare costs?",
                    "Are there any pharmaceutical shortages?",
                ],
                value="Select an option...",
                label="Select a query",
                interactive=True,
            )
            user_query = gr.Textbox(
                label="Enter your query",
                placeholder="E.g., Analyze the provided documents to identify factors that can impact medical inflation in the UK.",
                lines=4,
                visible=False,
            )
            selected_query_display = gr.Textbox(
                label="Selected Query",
                placeholder="Your selected query will appear here.",
                lines=2,
                interactive=False,
            )
            query_button = gr.Button("Submit Query")
            output_response = gr.Textbox(
                label="Response",
                placeholder="The response will appear here...",
                lines=10,
                interactive=False,
            )
            source_dropdown = gr.Dropdown(
                label="Sources",
                choices=[],
                interactive=True,
                visible=False,
            )
            source_display = gr.Textbox(
                label="Source Content",
                placeholder="Select a source to view its content.",
                lines=10,
                interactive=False,
            )

            # Query handling
            query_choices.change(
                fn=toggle_textbox,
                inputs=query_choices,
                outputs=[user_query, selected_query_display],
            )
            query_button.click(
                fn=handle_query,
                inputs=[
                    selected_query_display,
                    user_query,
                    data_folder_input_vis,
                    session_id,
                ],
                outputs=[output_response, source_dropdown],
            )
            source_dropdown.change(
                fn=handle_source_selection,
                inputs=[source_dropdown, data_folder_input_vis],
                outputs=source_display,
            )

if __name__ == "__main__":
    interface.launch()
