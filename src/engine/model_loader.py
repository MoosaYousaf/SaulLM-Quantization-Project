import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "Equall/Saul-7B-Instruct-v1"

SYSTEM_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
)

def get_8bit_config():
    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
    )

def get_4bit_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

def load_tokenizer(model_id=MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_fp16(model_id=MODEL_ID, device_map="auto"):
    print(f"[FP16] Loading in full FP16 ...")
    tokenizer = load_tokenizer(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device_map, trust_remote_code=True
    )
    model.eval()
    _print_param_count(model, "FP16")
    return model, tokenizer

def load_8bit(model_id=MODEL_ID, device_map="auto"):
    print(f"[8-bit] Loading with LLM.int8() quantization ...")
    tokenizer = load_tokenizer(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=get_8bit_config(),
        device_map=device_map, torch_dtype=torch.float16, trust_remote_code=True
    )
    model.eval()
    _print_param_count(model, "8-bit")
    return model, tokenizer

def load_4bit(model_id=MODEL_ID, device_map="auto"):
    print(f"[4-bit] Loading with NF4 4-bit quantization ...")
    tokenizer = load_tokenizer(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=get_4bit_config(),
        device_map=device_map, trust_remote_code=True
    )
    model.eval()
    _print_param_count(model, "4-bit")
    return model, tokenizer

def load_model_and_tokenizer(model_id=MODEL_ID, precision="8bit", device_map="auto"):
    precision = precision.lower().replace("-", "").replace("_", "")
    loaders = {"fp16": load_fp16, "8bit": load_8bit, "4bit": load_4bit}
    if precision not in loaders:
        raise ValueError(f"precision must be one of {list(loaders.keys())}, got '{precision}'")
    return loaders[precision](model_id=model_id, device_map=device_map)

def generate_response(model, tokenizer, user_prompt, max_new_tokens=256, do_sample=False, temperature=1.0):
    full_prompt = f"{SYSTEM_PROMPT}### Instruction:\n{user_prompt}\n\n### Response:"
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def _print_param_count(model, label):
    total = sum(p.numel() for p in model.parameters())
    print(f"[{label}] Model loaded -- total parameters: {total:,}")

def hello_world_test(precision="8bit"):
    model, tokenizer = load_model_and_tokenizer(precision=precision)
    prompt = "Hello! Can you briefly introduce yourself and explain what you are designed to help with?"
    print("\n" + "="*60)
    print(f"PROMPT ({precision}):\n{prompt}")
    print("="*60)
    response = generate_response(model, tokenizer, prompt, max_new_tokens=150)
    print(f"RESPONSE:\n{response}")
    print("="*60 + "\n")
    return response
