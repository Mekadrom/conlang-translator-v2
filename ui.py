import gradio as gr
import json
import pandas as pd
import requests

LLM_SERVER_URL = "http://localhost:8080"

GEN_ENDPOINT = "submit"

URL = f"{LLM_SERVER_URL}/{GEN_ENDPOINT}"

CONLANG_NAME = "Bakdila'abitz"

history = []
lexicon = []

def get_english_example(input):
    response = requests.post(URL, json={"prompt": input, "history": history})
    return response.json()["response"]

def get_example(prompt):
    return get_english_example(prompt)

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

# def save_lexicon_dataframe(lexicon_table):
#     global lexicon
#     lexicon = lexicon_table.values.tolist()
#     save_lexicon()
#     return lexicon

with gr.Blocks() as app:
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

        with gr.Blocks() as submission_form:
            with gr.Column():
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

app.launch()
