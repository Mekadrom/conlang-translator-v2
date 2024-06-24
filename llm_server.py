from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import json
import random

app = Flask(__name__)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct", 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

gen_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

classify_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 1.0,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
}

classification_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "do_sample": False,
    "temperature": 0.0,
}

gen_system_prompt = "You will generate random sentences in English so that a user can translate it into their conlang. The user will ask for simple, intermediate, and advanced sentences. Your output will be diverse but not too obscure. Only answer with a single example sentence and nothing more. Do not include any notes to the user."
gen_system_prompt_token_count = len(tokenizer(gen_system_prompt)["input_ids"])

classify_system_prompt = "You will classify the words in a sentence by their part of speech. For each word in the input sentence, repeat the word followed by the part of speech in parentheses. For example, 'The (DET) cat (NOUN) is (VERB) cute (ADJ).' Only answer with this format, and nothing more."
classify_system_prompt_token_count = len(tokenizer(classify_system_prompt)["input_ids"])

rlhf_system_prompt = "Given a history of sentences written in a user's conlang, their English translations, and the entire lexicon of the conlang, attempt to construct a new sentence in that conlang. Only answer with a single example sentence and nothing more. Do not include any notes to the user."
rlhf_system_prompt_token_count = len(tokenizer(rlhf_system_prompt)["input_ids"])

print(f"Generation system prompt token count: {gen_system_prompt_token_count}")
print(f"Classification system prompt token count: {classify_system_prompt_token_count}")
print(f"RLHF system prompt token count: {rlhf_system_prompt_token_count}")

with open('lexicon.json', 'r') as f:
    lexicon = json.load(f)

LEXICON_STRING = '\n'.join([f'"{conlang}": "{pos}"' for conlang, _, pos, _ in lexicon])

@app.route('/rlhf', methods=['GET'])
def rlhf():
    history = request.json["history"]

    messages = [
        {
            "role": "user",
            "content": rlhf_system_prompt,
        },
        {
            "role": "assistant",
            "content": "Sure, I can help with that. Please paste the history of conlang sentences and I will attempt to generate a new sentence.",
        },
        {
            "role": "user",
            "content": "Sentence history:\n" + '\n'.join([f'"{entry["response"]}": "{entry["example"]}"' for entry in history]) + '\nLexicon (words and their parts of speech):\n' + LEXICON_STRING,
        },
    ]

    response = gen_pipe(
        messages,
        **generation_args,
    )

    response = response[0]["generated_text"].strip()

    return jsonify({
        "response": response,
    })

@app.route('/submit', methods=['GET'])
def submit():
    history = request.json["history"]

    message_history = []

    for entry in history:
        message_history.append({
            "role": "user",
            "content": entry["prompt"],
        })
        message_history.append({
            "role": "assistant",
            "content": entry["example"],
        })

    # truncate history to last 10 entries
    message_history = message_history[::-1][:min(10, len(message_history))][::-1]
    print(message_history)

    messages = [
        {
            "role": "user",
            "content": gen_system_prompt,
        },
        {
            "role": "assistant",
            "content": "Sure, I can help with that. Which type of example sentence would you like?",
        },
        *message_history,
        {
            "role": "user",
            "content": request.json["prompt"],
        }
    ]

    response = gen_pipe(
        messages,
        **generation_args,
    )

    response = response[0]["generated_text"].strip()

    response = response.split("\n")[0]

    return jsonify({
        "response": response,
    })

@app.route('/classify', methods=['POST'])
def classify():
    messages = [
        {
            "role": "user",
            "content": classify_system_prompt,
        },
        {
            "role": "assistant",
            "content": "Sure, I can help with that. What sentence would you like me to classify the words of?",
        },
        {
            "role": "user",
            "content": "Classify: \"" + request.json["prompt"] + "\"",
        }
    ]

    response = classify_pipe(
        messages,
        **generation_args,
    )

    response = response[0]["generated_text"].strip()

    return jsonify({
        "response": response,
    })

@app.route('/arbitrary', methods=['POST'])
def arbitrary():
    messages = request.json["messages"]

    cur_gen_args = generation_args.copy()
    if "gen_args" in request.json:
        cur_gen_args.update(request.json["gen_args"])

    response = gen_pipe(
        messages,
        **generation_args,
    )

    response = response[0]["generated_text"].strip()

    return jsonify({
        "response": response,
    })

if __name__ == '__main__':
    app.run(port=8080)
