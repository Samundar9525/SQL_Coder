import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import argparse
import requests
import json

def generate_prompt(question, prompt_file="prompt.md", metadata_file="metadata.sql"):
    with open(prompt_file, "r") as f:
        prompt = f.read()
    
    with open(metadata_file, "r") as f:
        table_metadata_string = f.read()

    prompt = prompt.format(
        user_question=question, table_metadata_string=table_metadata_string
    )
    return prompt


def get_tokenizer_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=True,
    )
    return tokenizer, model

def run_inference(question, prompt_file="prompt.md", metadata_file="metadata.sql"):
    prompt = generate_prompt(question, prompt_file, metadata_file)
    # Ollama API endpoint
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "sqlcoder:15b",  # or whatever your model is called in Ollama
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 300,
            "stop": ["```"]
        }
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    result = response.json()
    generated_query = result["response"].split(";")[0].split("```")[0].strip() + ";"
    return generated_query

if __name__ == "__main__":
    # Parse arguments
    _default_question="Do we get more sales from customers in New York compared to customers in San Francisco? Give me the total sales for each city, and the difference between the two."
    parser = argparse.ArgumentParser(description="Run inference on a question")
    parser.add_argument("-q","--question", type=str, default=_default_question, help="Question to run inference on")
    args = parser.parse_args()
    question = args.question
    print("Loading a model and generating a SQL query for answering your question...")
    print(run_inference(question))