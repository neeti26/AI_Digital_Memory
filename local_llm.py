import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = "microsoft/Phi-3-mini-4k-instruct"
cache_path = "D:/hf_cache"

print("Loading model...")

# ---------------- Tokenizer ----------------
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_path
)

# ---------------- 4-bit Quantization ----------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# ---------------- Model ----------------
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="cuda",   # Force GPU
    cache_dir=cache_path
)

print("Model loaded successfully!")
print("Model device:", next(model.parameters()).device)


# ---------------- Generate Response ----------------
def generate_response(prompt, max_new_tokens=120):

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.15,          # Lower = less hallucination
            do_sample=True,
            top_p=0.75,               # Slightly tighter sampling
            repetition_penalty=1.2,   # Reduce looping
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt echo safely
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):]

    # If using "Final Answer:" anchor, extract clean output
    if "Final Answer:" in decoded:
        decoded = decoded.split("Final Answer:")[-1]

    return decoded.strip()