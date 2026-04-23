import csv
import gc
import json
import os
import sys
from typing import Dict, List

import torch

# Force Python to look at the parent directory to find 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.prompt_pipeline import format_legal_prompt
from src.engine.model_loader import generate_response, load_model_and_tokenizer
from src.evaluation.accuracy import score_nda_summary
from src.telemetry.metrics import PerformanceTracker

os.makedirs("outputs", exist_ok=True)
LATENCY_CSV_FILE = "outputs/metrics_log.csv"
ACCURACY_CSV_FILE = "outputs/accuracy_log.csv"
RESPONSES_TXT_FILE = "outputs/demo_responses.txt"
RESPONSES_JSON_FILE = "outputs/demo_responses.json"
MODEL_ID = "Equall/Saul-7B-Instruct-v1"

PRECISION_TO_LOADER = {
    "baseline": "fp16",
    "8-bit": "8bit",
    "4-bit": "4bit",
}


def _wipe_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def benchmark_model(precision: str, prompt: str) -> Dict[str, Dict[str, float]]:
    _wipe_memory()
    tracker = PerformanceTracker()

    print(f"\n[{precision.upper()}] Loading tokenizer and model...")
    model, tokenizer = load_model_and_tokenizer(
        model_id=MODEL_ID,
        precision=PRECISION_TO_LOADER[precision],
        device_map="auto",
    )

    print(f"[{precision.upper()}] Running inference phases...")

    tracker.start_phase("pre_processing")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    tracker.end_phase()

    tracker.start_phase("inference")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=220,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    tracker.end_phase()

    tracker.start_phase("post_processing")
    generated_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    tracker.end_phase()

    print(f"[{precision.upper()}] Run complete.")

    del model
    del tokenizer
    del inputs
    del output_ids
    _wipe_memory()

    phase_metrics = tracker.phases
    phase_metrics["response"] = {"text": decoded_text}
    return phase_metrics


def run_all_benchmarks() -> Dict[str, str]:
    nda_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'raw_documents', 'mock_nda.txt')

    try:
        prompt = format_legal_prompt(nda_path)
        print("✅ Successfully loaded Mock NDA for benchmarking.")
    except Exception:
        print("⚠️ Warning: Could not load mock_nda.txt. Falling back to default prompt.")
        prompt = (
            "Review this contract clause for indemnification liabilities: The Contractor agrees to indemnify and "
            "hold harmless the Client from any claims resulting from the Contractor's negligence."
        )

    # Run from smallest to largest to reduce OOM risk on smaller GPUs.
    precisions = ["4-bit", "8-bit", "baseline"]

    all_outputs: List[Dict[str, str]] = []

    with open(LATENCY_CSV_FILE, mode='w', newline='', encoding='utf-8') as latency_file, open(
        ACCURACY_CSV_FILE, mode='w', newline='', encoding='utf-8'
    ) as accuracy_file:
        latency_writer = csv.writer(latency_file)
        accuracy_writer = csv.writer(accuracy_file)

        latency_writer.writerow(["Precision", "Phase", "Time_sec", "Peak_Memory_MB"])
        accuracy_writer.writerow(
            [
                "Precision",
                "Accuracy",
                "ConfidentialInformationScore",
                "ObligationsScore",
                "GoverningLawScore",
            ]
        )

        for prec in precisions:
            run_data = benchmark_model(prec, prompt)
            response_text = run_data["response"]["text"]
            accuracy = score_nda_summary(response_text)

            for phase in ["pre_processing", "inference", "post_processing"]:
                metrics = run_data[phase]
                latency_writer.writerow(
                    [
                        prec,
                        phase,
                        f"{metrics['time_sec']:.4f}",
                        f"{metrics['peak_memory_mb']:.2f}",
                    ]
                )

            accuracy_writer.writerow(
                [
                    prec,
                    f"{accuracy['accuracy']:.4f}",
                    f"{accuracy['confidential_information_score']:.4f}",
                    f"{accuracy['obligations_receiving_party_score']:.4f}",
                    f"{accuracy['governing_law_score']:.4f}",
                ]
            )

            all_outputs.append(
                {
                    "precision": prec,
                    "accuracy": round(accuracy["accuracy"], 4),
                    "confidential_information_score": accuracy["confidential_information_score"],
                    "obligations_receiving_party_score": accuracy["obligations_receiving_party_score"],
                    "governing_law_score": accuracy["governing_law_score"],
                    "response": response_text,
                }
            )

    with open(RESPONSES_TXT_FILE, "w", encoding="utf-8") as txt_file:
        for item in all_outputs:
            txt_file.write("=" * 90 + "\n")
            txt_file.write(f"PRECISION: {item['precision']}\n")
            txt_file.write(f"ACCURACY: {item['accuracy']:.4f}\n")
            txt_file.write(
                "SCORES: "
                f"confidential={item['confidential_information_score']:.1f}, "
                f"obligations={item['obligations_receiving_party_score']:.1f}, "
                f"law={item['governing_law_score']:.1f}\n"
            )
            txt_file.write("RESPONSE:\n")
            txt_file.write(item["response"] + "\n\n")

    with open(RESPONSES_JSON_FILE, "w", encoding="utf-8") as json_file:
        json.dump(all_outputs, json_file, indent=2)

    print(f"\n🎯 Benchmarks completed! Latency log: {LATENCY_CSV_FILE}")
    print(f"🎯 Accuracy log: {ACCURACY_CSV_FILE}")
    print(f"🎯 Response demos: {RESPONSES_TXT_FILE} and {RESPONSES_JSON_FILE}")

    return {
        "latency_csv": LATENCY_CSV_FILE,
        "accuracy_csv": ACCURACY_CSV_FILE,
        "responses_txt": RESPONSES_TXT_FILE,
        "responses_json": RESPONSES_JSON_FILE,
    }


def main() -> None:
    run_all_benchmarks()


if __name__ == "__main__":
    main()
