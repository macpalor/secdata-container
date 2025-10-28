import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import os
import time

# -----------------------------
# Local model path
# -----------------------------
model_path = "./models/medgemma-27b-text-it"
adapter_save_path = "./lora-test-output"

print("Starting LoRA fine-tuning test.")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -----------------------------
# Load model and tokenizer (offline)
# -----------------------------
print("\nLoading model and tokenizer from local files...")
start_load = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
)
print(f"Model loaded in {time.time() - start_load:.2f} seconds")

# -----------------------------
# Apply LoRA configuration
# -----------------------------
print("\nAttaching LoRA adapters...")
lora_config = LoraConfig(
    r=4,              # rank of the adaptation matrices (tiny for testing)
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # common for transformer attention
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
print("LoRA adapters added successfully.")

# -----------------------------
# Dummy training data
# -----------------------------
prompt = "La medicina moderna utilizza l'intelligenza artificiale."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Prepare dummy labels (same as inputs for simplicity)
labels = inputs["input_ids"].clone()

# -----------------------------
# Simple training loop
# -----------------------------
print("\nRunning dummy training step...")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

start_train = time.time()
for step in range(3):
    optimizer.zero_grad()
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f"Step {step+1}/3 - Loss: {loss.item():.4f}")

train_time = time.time() - start_train
print(f"Dummy training completed in {train_time:.2f} seconds")

# -----------------------------
# Save adapter weights
# -----------------------------
print("\nSaving LoRA adapter weights...")
os.makedirs(adapter_save_path, exist_ok=True)
model.save_pretrained(adapter_save_path)
print(f"Adapters saved to: {adapter_save_path}")

if device == "cuda":
    print(f"Final GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

print("\nLoRA fine-tuning test completed successfully.")