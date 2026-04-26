import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
        llm_int8_enable_fp32_cpu_offload=False,
    )


def get_4bit_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def _get_default_max_memory(max_gpu_memory_gib: str = "13GiB", max_cpu_memory_gib: str = "48GiB"):
    if torch.cuda.is_available():
        return {0: max_gpu_memory_gib, "cpu": max_cpu_memory_gib}
    return {"cpu": max_cpu_memory_gib}


def _common_model_kwargs(
    device_map="auto",
    max_memory=None,
    offload_folder="offload",
    enable_offload=False,
):
    kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if max_memory is not None:
        kwargs["max_memory"] = max_memory

    if enable_offload:
        os.makedirs(offload_folder, exist_ok=True)
        kwargs["offload_folder"] = offload_folder
        kwargs["offload_state_dict"] = True

    return kwargs


def load_tokenizer(model_id=MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_fp16(model_id=MODEL_ID, device_map="auto", max_memory=None, offload_folder="offload"):
    print("[FP16] Loading FP16 with CPU offload safeguards ...", flush=True)
    tokenizer = load_tokenizer(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        **_common_model_kwargs(
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
            enable_offload=True,
        ),
    )
    model.eval()
    model.config.use_cache = False
    _print_param_count(model, "FP16")
    return model, tokenizer


def load_8bit(model_id=MODEL_ID, device_map="auto", max_memory=None, offload_folder="offload"):
    print("[8-bit] Loading with LLM.int8() (GPU-first, no state-dict offload) ...", flush=True)
    tokenizer = load_tokenizer(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=get_8bit_config(),
        torch_dtype=torch.float16,
        **_common_model_kwargs(
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
            enable_offload=False,
        ),
    )
    model.eval()
    model.config.use_cache = False
    _print_param_count(model, "8-bit")
    return model, tokenizer


def load_4bit(model_id=MODEL_ID, device_map="auto", max_memory=None, offload_folder="offload"):
    print("[4-bit] Loading NF4 (GPU-first, no state-dict offload) ...", flush=True)
    tokenizer = load_tokenizer(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=get_4bit_config(),
        torch_dtype=torch.float16,
        **_common_model_kwargs(
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
            enable_offload=False,
        ),
    )
    model.eval()
    model.config.use_cache = False
    _print_param_count(model, "4-bit")
    return model, tokenizer


def load_model_and_tokenizer(
    model_id=MODEL_ID,
    precision="8bit",
    device_map="auto",
    max_memory=None,
    offload_folder="offload",
):
    precision = precision.lower().replace("-", "").replace("_", "")
    loaders = {"fp16": load_fp16, "8bit": load_8bit, "4bit": load_4bit}
    if precision not in loaders:
        raise ValueError(f"precision must be one of {list(loaders.keys())}, got '{precision}'")

    if max_memory is None:
        max_memory = _get_default_max_memory()

    return loaders[precision](
        model_id=model_id,
        device_map=device_map,
        max_memory=max_memory,
        offload_folder=offload_folder,
    )


def generate_response(model, tokenizer, user_prompt, max_new_tokens=128, do_sample=False, temperature=1.0):
    full_prompt = f"{SYSTEM_PROMPT}### Instruction:\n{user_prompt}\n\n### Response:"
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _print_param_count(model, label):
    total = sum(p.numel() for p in model.parameters())
    print(f"[{label}] Model loaded -- total parameters: {total:,}", flush=True)


def hello_world_test(precision="8bit"):
    model, tokenizer = load_model_and_tokenizer(precision=precision)
    prompt = "Hello! Can you briefly introduce yourself and explain what you are designed to help with?"
    print("\n" + "=" * 60)
    print(f"PROMPT ({precision}):\n{prompt}")
    print("=" * 60)
    response = generate_response(model, tokenizer, prompt, max_new_tokens=100)
    print(f"RESPONSE:\n{response}")
    print("=" * 60 + "\n")
    return response
