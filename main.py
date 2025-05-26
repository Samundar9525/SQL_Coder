import argparse
from ctransformers import AutoModelForCausalLM, AutoTokenizer
import psycopg2
import time
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any

app = FastAPI()

def generate_prompt(question, prompt_file="prompt.md", metadata_file="metadata.sql", max_metadata_chars=400, max_chars=1200):
    with open(prompt_file, "r") as f:
        prompt = f.read()
    with open(metadata_file, "r") as f:
        table_metadata_string = f.read()
    # Truncate metadata to avoid exceeding context window
    if len(table_metadata_string) > max_metadata_chars:
        table_metadata_string = table_metadata_string[:max_metadata_chars] + "\n-- METADATA TRUNCATED --"
    prompt = prompt.format(
        user_question=question, table_metadata_string=table_metadata_string
    )
    # Fallback: truncate by character count
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars]
    return prompt

def run_inference(question, prompt_file="prompt.md", metadata_file="metadata.sql"):
    model_path = r"C:\Users\samun\Downloads\sqlcoder-7b-2.Q2_K.gguf"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        model_type="llama",
        max_new_tokens=300,
        gpu_layers=0,
    )
    prompt = generate_prompt(question, prompt_file, metadata_file)
    print("Generated SQL (streaming):")
    generated_query = ""
    for chunk in model(prompt, stop=["```", ";"], stream=True):
        print(chunk, end="", flush=True)
        generated_query += chunk
    print()  # for newline after streaming
    # Optionally, clean up the output
    generated_query = generated_query.split(";")[0].split("```")[0].strip() + ";"
    return generated_query

def execute_query(sql, db_params):
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        cur.execute(sql)
        result = cur.fetchall()
        cur.close()
        conn.close()
        return result
    except Exception as e:
        print("Database error:", e)
        return None

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QuestionRequest):
    db_params = {
        "host": "localhost",
        "port": 5432,
        "database": "employee",
        "user": "postgres",
        "password": "admin"
    }
    max_retries = 5
    attempt = 0
    result = None
    sql_query = ""
    while attempt < max_retries:
        sql_query = run_inference(request.question)
        result = execute_query(sql_query, db_params)
        if result and len(result) > 0:
            break
        attempt += 1
    return {
        "question": request.question,
        "sql_query": sql_query,
        "result": result if result else [],
        "attempts": attempt + 1
    }

@app.post("/ask_stream")
def ask_question_stream(request: QuestionRequest):
    def stream_generator():
        model_path = r"C:\Users\samun\Downloads\sqlcoder-7b-2.Q2_K.gguf"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="llama",
            max_new_tokens=300,
            gpu_layers=0,
        )
        prompt = generate_prompt(request.question)
        for chunk in model(prompt, stop=["```", ";"], stream=True):
            yield chunk  # Send each chunk as it is generated

    return StreamingResponse(stream_generator(), media_type="text/plain")

if __name__ == "__main__":
    _default_question = "show top 5 female employee "
    parser = argparse.ArgumentParser(description="Run inference on a question")
    parser.add_argument("-q", "--question", type=str, default=_default_question, help="Question to run inference on")
    args = parser.parse_args()
    question = args.question

    db_params = {
        "host": "localhost",
        "port": 5432,
        "database": "employee",
        "user": "postgres",
        "password": "admin"
    }

    max_retries = 5
    attempt = 0
    result = None

    while attempt < max_retries:
        print(f"\nAttempt {attempt + 1}: Loading model and generating SQL query...")
        sql_query = run_inference(question)
        print("\nGenerated SQL Query:\n", sql_query)
        print("\nQuery Result:")
        result = execute_query(sql_query, db_params)
        print(result)
        if result and len(result) > 0:
            break
        print("No result found, retrying...\n")
        attempt += 1
        time.sleep(1)  # Optional: wait 1 second before retry

    if not result or len(result) == 0:
        print("No results found after maximum retries.")