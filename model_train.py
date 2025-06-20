"""
This script demonstrates how to fine-tune a Hugging Face LLM (e.g., Llama-2, Mistral) for SQL generation
using your own dataset in CSV format with 'question' and 'sql_query' columns.

Requirements:
- pip install transformers datasets peft bitsandbytes accelerate

Example CSV:
question,sql_query
Describe the purpose of from_date.,SELECT from_date FROM title;
Which table contains dept_no?,SELECT dept_no FROM dept_emp;
Where can I see the first_name?,SELECT first_name FROM employee;
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Configurations
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Or any other Hugging Face model
DATA_PATH = "text2sql_dataset.csv"       # Your CSV dataset path
OUTPUT_DIR = "./finetuned-sql-model"
NUM_EPOCHS = 2
BATCH_SIZE = 1
MAX_LENGTH = 128

# Load dataset from CSV
dataset = load_dataset("csv", data_files=DATA_PATH, split="train")

# Preprocess: concatenate question and sql_query for instruction tuning
def preprocess(example):
    prompt = f"### Instruction:\n{example['question']}\n### Response:\n{example['sql_query']}"
    return {"text": prompt}

dataset = dataset.map(preprocess)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto", device_map="cpu")

# Optional: LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Tokenize
def tokenize_function(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    save_steps=100,
    save_total_limit=1,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=False,  # No fp16 on CPU
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Train
trainer.train()

# Save final model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Training complete. Model saved to {OUTPUT_DIR}")