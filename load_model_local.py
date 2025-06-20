import pandas as pd
import psycopg2
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("e:/QLGPT/SQL_Coder/merged-sql-model")
tokenizer = AutoTokenizer.from_pretrained("e:/QLGPT/SQL_Coder/merged-sql-model")

def generate_sql(question):
    prompt = f"### Instruction:\n{question}\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        eos_token_id=tokenizer.eos_token_id
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sql = answer.split("### Response:")[-1].strip()
    return sql

def execute_query(sql, conn):
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            result = cur.fetchall()
        return result
    except Exception as e:
        print(f"Query failed: {e}")
        return None

# Connect to Postgres
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="employee",
    user="postgres",
    password="admin"
)

# Load test data
df = pd.read_csv("test.csv")

# Evaluate
correct = 0
total = len(df)
for idx, row in df.iterrows():
    pred_sql = generate_sql(row["question"])
    gt_sql = row["sql_query"].strip()
    pred_result = execute_query(pred_sql, conn)
    gt_result = execute_query(gt_sql, conn)
    print(f"Q: {row['question']}")
    print(f"Expected SQL: {gt_sql}")
    print(f"Predicted SQL: {pred_sql}")
    print(f"Expected Result: {gt_result}")
    print(f"Predicted Result: {pred_result}")
    print("-" * 40)
    if pred_result == gt_result:
        correct += 1

accuracy = correct / total * 100
print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")

conn.close()