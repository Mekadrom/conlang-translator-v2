import gradio as gr
import json
import pandas as pd
import requests

LLM_SERVER_URL = "http://localhost:8080"

GEN_ENDPOINT = "submit"

GEN_URL = f"{LLM_SERVER_URL}/{GEN_ENDPOINT}"
RLHF_URL = f"{LLM_SERVER_URL}/rlhf"

CONLANG_NAME = "Bakdila'abitz"

history = []
lexicon = []

def get_english_example(input):
    response = requests.get(GEN_URL, json={"prompt": input, "history": history})
    return response.json()["response"]

def get_example(prompt):
    return get_english_example(prompt)

def get_rlhf_example():
    response = requests.get(RLHF_URL, json={"history": history})
    return response.json()["response"]

def submit_and_retrieve_rlhf(rlhf_tb, english_tb):
    global history
    history.append({"prompt": "rlhf", "example": english_tb, "response": rlhf_tb})
    return get_rlhf_example(), ""

def reject_and_retrieve_rlhf(rlhf_tb):
    return get_rlhf_example(), ""

def load_history():
    global history
    try:
        with open("history.json", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                history.append({"prompt": data["prompt"], "example": data["example"], "response": data["response"]})
    except FileNotFoundError:
        print("No history found. Starting fresh.")

def load_lexicon():
    global lexicon
    try:
        try:
            with open("lexicon.json", "r", encoding="utf-8") as f:
                lexicon = json.load(f)
        except json.JSONDecodeError:
            print("Lexicon is not in correct JSON format.")
    except FileNotFoundError:
        print("No lexicon found. Starting fresh.")

load_history()
load_lexicon()

def save_history():
    global history
    with open("history.json", "w", encoding="utf-8") as f:
        print(history)
        for d in history:
            f.write(json.dumps({"prompt": d["prompt"], "example": d["example"], "response": d["response"]}) + "\n")

def save_lexicon():
    global lexicon
    with open("lexicon.json", "w", encoding="utf-8") as f:
        json.dump(lexicon, f)

with gr.Blocks() as app:
    with gr.Tab("Data Entry"):
        with gr.Row(equal_height=False):
            with gr.Blocks() as submission_form:
                with gr.Column():
                    dropdown = gr.Dropdown(label="Difficulty", choices=["simple", "intermediate", "advanced"], value="simple", multiselect=False)
                    refresh = gr.Button("Refresh", variant="secondary")
                    example_tb = gr.Textbox(label=f"Translate the following sentence to {CONLANG_NAME}", show_copy_button=True, value=get_example("simple"), interactive=False)

                    response_tb = gr.Textbox(label="Response")
                    submit = gr.Button("Submit", variant="primary")

                    def submit_response(example, response):
                        global history
                        prompt = "simple"
                        history.append({"prompt": prompt, "example": example, "response": response})
                        save_history()

                        return get_example(prompt), ""

                    submit.click(fn=submit_response, inputs=[example_tb, response_tb], outputs=[example_tb, response_tb], api_name="submit_response")
                    refresh.click(fn=lambda val: get_example(val), inputs=[dropdown], outputs=[example_tb], api_name="refresh_example")
    with gr.Tab("RLHF"):
        with gr.Row(equal_height=False):
            with gr.Blocks() as lexicon_view:
                with gr.Column(variant="compact"):
                    # accepts json or csv
                    lexicon_file = gr.File(label="Lexicon File", type="filepath", file_types=['csv'])
                    lexicon_table = gr.Dataframe(value=lexicon, label="Lexicon Table", headers=[CONLANG_NAME, "English", "Part of Speech", "Description"], interactive=True, col_count=(4, "fixed"))

                    def load_lexicon_file(lexicon_file):
                        global lexicon

                        pd_dataframe = pd.read_csv(lexicon_file, encoding="utf-8", delimiter='|', keep_default_na=False)

                        lexicon = list(pd_dataframe.itertuples(index=False, name=None))
                        print(lexicon)

                        save_lexicon()
                        return lexicon
                    
                    lexicon_file.change(load_lexicon_file, inputs=[lexicon_file], outputs=[lexicon_table], api_name="load_lexicon_file")
            with gr.Blocks() as rlhf_view:
                with gr.Column(variant="compact"):
                    rlhf_tb = gr.Textbox(label="RLHF", show_copy_button=True, value=get_rlhf_example(), interactive=False)
                    english_tb = gr.Textbox(label="English", show_copy_button=True, value="", interactive=True)

                    rlhf_accept = gr.Button("👍", variant="primary")
                    rlhf_reject = gr.Button("👎", variant="secondary")

                    rlhf_accept.click(fn=submit_and_retrieve_rlhf, inputs=[rlhf_tb, english_tb], outputs=[rlhf_tb, english_tb], api_name="submit_rlhf")
                    rlhf_reject.click(fn=reject_and_retrieve_rlhf, inputs=[rlhf_tb], outputs=[rlhf_tb, english_tb], api_name="reject_rlhf")

app.launch()
