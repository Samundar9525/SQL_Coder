from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
peft_model = PeftModel.from_pretrained(base_model, "./finetuned-sql-model")
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("./merged-sql-model")