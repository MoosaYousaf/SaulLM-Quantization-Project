import sys
import os
import csv
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Force Python to look at the parent directory to find 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.telemetry.metrics import PerformanceTracker
from src.data.prompt_pipeline import format_legal_prompt

os.makedirs("outputs", exist_ok=True)
CSV_FILE = "outputs/metrics_log.csv"
MODEL_ID = "Equall/Saul-7B-Instruct-v1"

def benchmark_model(precision: str, prompt: str):
    # Initial memory wipe to ensure a clean slate
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tracker = PerformanceTracker()
    
    print(f"\n[{precision.upper()}] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Set pad token to avoid generation warnings
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------------------------------------------------
    # MODEL LOADING BLOCK (Fully Optimized for Colab)
    # ---------------------------------------------------------
    if precision == "baseline":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.float16, 
            device_map="auto",
            low_cpu_mem_usage=True
        )
    elif precision == "8-bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            quantization_config=quantization_config, 
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
    elif precision == "4-bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            quantization_config=quantization_config, 
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
    else:
        raise ValueError("Unsupported precision.")

    print(f"[{precision.upper()}] Running inference phases...")

    tracker.start_phase("pre_processing")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    tracker.end_phase()

    tracker.start_phase("inference")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    tracker.end_phase()

    tracker.start_phase("post_processing")
    decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tracker.end_phase()
    
    print(f"[{precision.upper()}] Run complete.")
    
    # ---------------------------------------------------------
    # SINGLE AGGRESSIVE MEMORY WIPING BLOCK
    # ---------------------------------------------------------
    del model
    del tokenizer
    del inputs
    if 'outputs' in locals():
        del outputs
        
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # ---------------------------------------------------------

    return tracker.phases

def main():
    # Attempt to load the realistic Mock NDA for the presentation
    nda_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'raw_documents', 'mock_nda.txt')
    
    try:
        prompt = format_legal_prompt(nda_path)
        print("✅ Successfully loaded Mock NDA for benchmarking.")
    except Exception as e:
        print("⚠️ Warning: Could not load mock_nda.txt. Falling back to default prompt.")
        prompt = "Review this contract clause for indemnification liabilities: The Contractor agrees to indemnify and hold harmless the Client from any claims resulting from the Contractor's negligence."

    # CRITICAL: Run from smallest to largest to prevent OOM crash on Colab T4
    precisions = ["4-bit", "8-bit", "baseline"]
    
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Precision", "Phase", "Time_sec", "Peak_Memory_MB"])
        
        for prec in precisions:
            phases_metrics = benchmark_model(prec, prompt)
            for phase, metrics in phases_metrics.items():
                writer.writerow([
                    prec, 
                    phase, 
                    f"{metrics['time_sec']:.4f}", 
                    f"{metrics['peak_memory_mb']:.2f}"
                ])
                
    print(f"\n🎯 All benchmarks completed successfully! Log saved to {CSV_FILE}.")

if __name__ == "__main__":
    main()
