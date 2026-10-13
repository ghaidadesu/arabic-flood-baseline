"""
Gradio interface for the single-agent baseline.

    python app.py            # opens on http://127.0.0.1:7860
    python app.py --share    # also creates a temporary public link (needed in Colab)

Warning: with --share anyone who has the link can use YOUR API quota.
Stop the app when the demo is over.
"""

import argparse

import baseline

EXAMPLES = [
    ["تحذير من السيول في جدة، يرجى الابتعاد عن مجاري الأودية"],
    ["اللهم احفظ أهلنا في الليث واجعلها سقيا رحمة"],
    ["مباراة اليوم كانت ممتعة جدا والليث كان الأفضل"],
]


def predict(tweet_text):
    """Returns (relevance, information type, details). Raises on problems."""
    result = baseline.classify_tweet(tweet_text)          # ValueError if the box is empty
    if not result["valid"]:
        raise RuntimeError(result["error"] or "The model did not return a valid answer.")
    details = (
        f"API calls: {result['calls']} | repairs: {result['repairs']} | "
        f"tokens (in / out / reasoning): {result['input_tokens']} / "
        f"{result['output_tokens']} / {result['reasoning_tokens']} | "
        f"time: {result['latency_s']:.1f} s"
    )
    return result["relevance"], result["information_type"], details


def build_ui():
    import gradio as gr

    def on_click(tweet_text):
        try:
            return predict(tweet_text)
        except (ValueError, RuntimeError) as error:
            raise gr.Error(str(error))

    with gr.Blocks(title="Arabic Flood Tweet Classifier") as demo:
        gr.Markdown(
            "# Arabic Flood Tweet Classifier\n"
            "### Single-Agent Baseline (3 Gemini API calls)\n"
            "Enter an Arabic tweet to get its relevance to a flood crisis and its information type."
        )
        tweet_input = gr.Textbox(label="Arabic Tweet", lines=4,
                                 placeholder="اكتب التغريدة العربية هنا...")
        button = gr.Button("Classify Tweet", variant="primary")
        with gr.Row():
            relevance_out = gr.Textbox(label="Relevance", interactive=False)
            info_out = gr.Textbox(label="Information Type", interactive=False)
        details_out = gr.Textbox(label="Run details", interactive=False)
        gr.Examples(examples=EXAMPLES, inputs=tweet_input)
        button.click(on_click, inputs=tweet_input,
                     outputs=[relevance_out, info_out, details_out])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="create a temporary public link")
    args = parser.parse_args()
    build_ui().launch(share=args.share)
