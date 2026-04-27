import argparse
import csv
import gc
import json
import os
import sys
from typing import Dict, List, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.prompt_pipeline import format_legal_prompt
from src.engine.model_loader import load_model_and_tokenizer
from src.evaluation.accuracy import score_nda_summary
from src.telemetry.metrics import PerformanceTracker

os.makedirs("outputs", exist_ok=True)
LATENCY_CSV_FILE = "outputs/metrics_log.csv"
ACCURACY_CSV_FILE = "outputs/accuracy_log.csv"
RESPONSES_TXT_FILE = "outputs/demo_responses.txt"
RESPONSES_JSON_FILE = "outputs/demo_responses.json"
DEFAULT_MODEL_ID = "Equall/Saul-7B-Instruct-v1"

PRECISION_TO_LOADER = {"baseline": "fp16", "16-bit": "fp16", "8-bit": "8bit", "4-bit": "4bit"}
PRECISION_ORDER = {"4-bit": 0, "8-bit": 1, "16-bit": 2}


def _normalize_precision_name(raw_precision: str) -> str:
    normalized = raw_precision.strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    aliases = {
        "4bit": "4-bit",
        "8bit": "8-bit",
        "16bit": "16-bit",
        "fp16": "16-bit",
        "baseline": "16-bit",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported precision '{raw_precision}'. Use one of: 4-bit, 8-bit, 16-bit (or baseline/fp16).")
    return aliases[normalized]


def _wipe_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _resolve_max_memory(max_gpu_memory: str, max_cpu_memory: str):
    if torch.cuda.is_available():
        return {0: max_gpu_memory, "cpu": max_cpu_memory}
    return {"cpu": max_cpu_memory}


def _precision_memory_caps(precision: str, max_gpu_memory: str, max_cpu_memory: str):
    if precision in {"4-bit", "8-bit"}:
        # Keep quantized loads GPU-first to avoid massive CPU RAM pressure in Colab.
        if torch.cuda.is_available():
            return {0: max_gpu_memory}
        return {"cpu": max_cpu_memory}
    return _resolve_max_memory(max_gpu_memory=max_gpu_memory, max_cpu_memory=max_cpu_memory)


def _get_device_for_inputs(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def benchmark_model(
    model_id: str,
    precision: str,
    prompt: str,
    max_new_tokens: int,
    max_input_tokens: int,
    max_memory: Dict,
    offload_folder: str,
) -> Tuple[Dict[str, Dict[str, float]], str]:
    _wipe_memory()
    tracker = PerformanceTracker()

    print(f"\n[{precision.upper()}] Loading tokenizer and model...", flush=True)
    device_map = {"": 0} if torch.cuda.is_available() and precision in {"4-bit", "8-bit"} else "auto"
    model, tokenizer = load_model_and_tokenizer(
        model_id=model_id,
        precision=PRECISION_TO_LOADER[precision],
        device_map=device_map,
        max_memory=max_memory,
        offload_folder=offload_folder,
    )

    print(f"[{precision.upper()}] Running inference phases...", flush=True)

    tracker.start_phase("pre_processing")
    input_device = _get_device_for_inputs(model)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(input_device)
    tracker.end_phase()

    tracker.start_phase("inference")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    tracker.end_phase()

    tracker.start_phase("post_processing")
    generated_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    tracker.end_phase()

    print(f"[{precision.upper()}] Run complete.", flush=True)

    del model, tokenizer, inputs, output_ids
    _wipe_memory()

    return tracker.phases, decoded_text


def run_all_benchmarks(
    model_id: str,
    precisions: List[str],
    max_new_tokens: int,
    max_input_tokens: int,
    max_gpu_memory: str,
    max_cpu_memory: str,
    offload_folder: str,
) -> Dict[str, str]:
    nda_path = os.path.join(os.path.dirname(__file__), "..", "src", "data", "raw_documents", "mock_nda.txt")

    try:
        prompt = format_legal_prompt(nda_path)
        print("✅ Successfully loaded Mock NDA for benchmarking.", flush=True)
    except Exception:
        print("⚠️ Warning: Could not load mock_nda.txt. Falling back to default prompt.", flush=True)
        prompt = (
            "Review this contract clause for indemnification liabilities: The Contractor agrees to indemnify and "
            "hold harmless the Client from any claims resulting from the Contractor's negligence."
        )

    all_outputs: List[Dict[str, str]] = []

    with open(LATENCY_CSV_FILE, mode="w", newline="", encoding="utf-8") as latency_file, open(
        ACCURACY_CSV_FILE, mode="w", newline="", encoding="utf-8"
    ) as accuracy_file:
        latency_writer = csv.writer(latency_file)
        accuracy_writer = csv.writer(accuracy_file)

        latency_writer.writerow(["Precision", "Status", "Phase", "Time_sec", "Peak_Memory_MB"])
        accuracy_writer.writerow(
            ["Precision", "Status", "Accuracy", "ConfidentialInformationScore", "ObligationsScore", "GoverningLawScore", "Error"]
        )

        for prec in precisions:
            max_memory = _precision_memory_caps(prec, max_gpu_memory=max_gpu_memory, max_cpu_memory=max_cpu_memory)
            try:
                phase_metrics, response_text = benchmark_model(
                    model_id=model_id,
                    precision=prec,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    max_input_tokens=max_input_tokens,
                    max_memory=max_memory,
                    offload_folder=offload_folder,
                )
                accuracy = score_nda_summary(response_text)

                for phase in ["pre_processing", "inference", "post_processing"]:
                    metrics = phase_metrics[phase]
                    latency_writer.writerow([prec, "ok", phase, f"{metrics['time_sec']:.4f}", f"{metrics['peak_memory_mb']:.2f}"])

                accuracy_writer.writerow(
                    [
                        prec,
                        "ok",
                        f"{accuracy['accuracy']:.4f}",
                        f"{accuracy['confidential_information_score']:.4f}",
                        f"{accuracy['obligations_receiving_party_score']:.4f}",
                        f"{accuracy['governing_law_score']:.4f}",
                        "",
                    ]
                )

                all_outputs.append(
                    {
                        "precision": prec,
                        "status": "ok",
                        "accuracy": round(accuracy["accuracy"], 4),
                        "confidential_information_score": accuracy["confidential_information_score"],
                        "obligations_receiving_party_score": accuracy["obligations_receiving_party_score"],
                        "governing_law_score": accuracy["governing_law_score"],
                        "response": response_text,
                    }
                )

            except (torch.cuda.OutOfMemoryError, RuntimeError) as load_error:
                message = str(load_error).replace("\n", " ")
                if "out of memory" not in message.lower() and "cuda" not in message.lower():
                    raise

                print(f"❌ [{prec}] OOM/Runtime memory error: {message}", flush=True)
                latency_writer.writerow([prec, "oom", "pre_processing", "", ""])
                latency_writer.writerow([prec, "oom", "inference", "", ""])
                latency_writer.writerow([prec, "oom", "post_processing", "", ""])
                accuracy_writer.writerow([prec, "oom", "", "", "", "", message])

                all_outputs.append({"precision": prec, "status": "oom", "accuracy": None, "response": "", "error": message})
                _wipe_memory()
                continue
            except Exception as run_error:
                message = str(run_error).replace("\n", " ")
                print(f"❌ [{prec}] Failed: {message}", flush=True)
                latency_writer.writerow([prec, "error", "pre_processing", "", ""])
                latency_writer.writerow([prec, "error", "inference", "", ""])
                latency_writer.writerow([prec, "error", "post_processing", "", ""])
                accuracy_writer.writerow([prec, "error", "", "", "", "", message])
                all_outputs.append({"precision": prec, "status": "error", "accuracy": None, "response": "", "error": message})
                _wipe_memory()
                continue
            except KeyboardInterrupt:
                print(f"⚠️ Interrupted during precision '{prec}'. Writing partial outputs before exit.", flush=True)
                break

    with open(RESPONSES_TXT_FILE, "w", encoding="utf-8") as txt_file:
        for item in all_outputs:
            txt_file.write("=" * 90 + "\n")
            txt_file.write(f"PRECISION: {item['precision']}\n")
            txt_file.write(f"STATUS: {item['status']}\n")
            if item["status"] == "ok":
                txt_file.write(f"ACCURACY: {item['accuracy']:.4f}\n")
                txt_file.write(
                    "SCORES: "
                    f"confidential={item['confidential_information_score']:.1f}, "
                    f"obligations={item['obligations_receiving_party_score']:.1f}, "
                    f"law={item['governing_law_score']:.1f}\n"
                )
                txt_file.write("RESPONSE:\n")
                txt_file.write(item["response"] + "\n\n")
            else:
                txt_file.write("ERROR:\n")
                txt_file.write(item.get("error", "Unknown error") + "\n\n")

    with open(RESPONSES_JSON_FILE, "w", encoding="utf-8") as json_file:
        json.dump(all_outputs, json_file, indent=2)

    print(f"\n🎯 Benchmarks completed! Latency log: {LATENCY_CSV_FILE}", flush=True)
    print(f"🎯 Accuracy log: {ACCURACY_CSV_FILE}", flush=True)
    print(f"🎯 Response demos: {RESPONSES_TXT_FILE} and {RESPONSES_JSON_FILE}", flush=True)

    return {
        "latency_csv": LATENCY_CSV_FILE,
        "accuracy_csv": ACCURACY_CSV_FILE,
        "responses_txt": RESPONSES_TXT_FILE,
        "responses_json": RESPONSES_JSON_FILE,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quantization latency + accuracy benchmark for SaulLM.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id to benchmark.")
    parser.add_argument(
        "--precisions",
        default="4-bit,8-bit,16-bit",
        help="Comma-separated precisions to run. Run order is always 4-bit -> 8-bit -> 16-bit.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Max generated tokens per run.")
    parser.add_argument("--max-input-tokens", type=int, default=2048, help="Tokenizer truncation cap for input prompt.")
    parser.add_argument("--max-gpu-memory", default="12GiB", help="GPU memory cap for placement.")
    parser.add_argument("--max-cpu-memory", default="48GiB", help="CPU RAM cap for offloading.")
    parser.add_argument("--offload-folder", default="offload", help="Folder for CPU/disk offloaded weights.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalized_precisions = [_normalize_precision_name(p) for p in args.precisions.split(",") if p.strip()]
    precisions = sorted(set(normalized_precisions), key=lambda p: PRECISION_ORDER[p])
    print(f"🔁 Precision run order: {', '.join(precisions)}", flush=True)

    run_all_benchmarks(
        model_id=args.model_id,
        precisions=precisions,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        max_gpu_memory=args.max_gpu_memory,
        max_cpu_memory=args.max_cpu_memory,
        offload_folder=args.offload_folder,
    )


if __name__ == "__main__":
    main()
