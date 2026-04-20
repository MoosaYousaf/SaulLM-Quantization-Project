import os
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from src.telemetry.metrics import PerformanceTracker

os.makedirs("outputs", exist_ok=True)
CSV_FILE = "outputs/metrics_log.csv"

def benchmark_model(precision: str, prompt: str):
    tracker = PerformanceTracker()
    model_id = "Equall/SaulLM-7B" 
    
    print(f"\n[{precision.upper()}] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if precision == "baseline":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
    elif precision == "8-bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            quantization_config=quantization_config, 
            device_map="auto"
        )
    elif precision == "4-bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            quantization_config=quantization_config, 
            device_map="auto"
        )
    else:
        raise ValueError("Unsupported precision.")

    tracker.start_phase("pre_processing")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    tracker.end_phase()

    tracker.start_phase("inference")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    tracker.end_phase()

    tracker.start_phase("post_processing")
    decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tracker.end_phase()
    
    print(f"[{precision.upper()}] Run complete.")
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return tracker.phases

def main():
    prompt = "Review this contract clause for indemnification liabilities: The Contractor agrees to indemnify and hold harmless the Client from any claims resulting from the Contractor's negligence."
    precisions = ["baseline", "8-bit", "4-bit"]
    
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
                
    print(f"\nLog saved to {CSV_FILE}.")

if __name__ == "__main__":
    main()
