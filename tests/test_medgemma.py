import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Local model path (adjust as needed)
# -----------------------------
model_path = "./models/medgemma-27b-text-it"

print("Offline MedGemma test started.")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Initial GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"Initial GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

# -----------------------------
# Load model and tokenizer (offline)
# -----------------------------
print("\nLoading model and tokenizer from local files...")
start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
)
load_time = time.time() - start_time
print(f"Model loaded in {load_time:.2f} seconds")

if device == "cuda":
    print(f"After load - GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"After load - GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

# -----------------------------
# Run a small inference test
# -----------------------------
prompt = "Riassumi in una frase: L'intelligenza artificiale sta trasformando la medicina moderna."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("\nRunning inference...")
start_infer = time.time()
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50)
infer_time = time.time() - start_infer
print(f"Inference completed in {infer_time:.2f} seconds")

# Decode output
result = tokenizer.decode(output[0], skip_special_tokens=True)
print("\n--- Model Output ---")
print(result)

if device == "cuda":
    torch.cuda.synchronize()
    print(f"\nFinal GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"Final GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

print("\nOffline MedGemma test completed successfully.")