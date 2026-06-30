# SaulLM Quantization Project (16-bit vs 8-bit vs 4-bit)

An end-to-end benchmarking harness that measures how **post-training quantization** affects the
latency, memory footprint, and output quality of a 7-billion-parameter legal language model
(**`Equall/Saul-7B-Instruct-v1`**) on a real legal task: summarizing the key clauses of a
Non-Disclosure Agreement (NDA).

In a single command the project loads the model at three precisions — **16-bit (FP16)**,
**8-bit (LLM.int8())**, and **4-bit (NF4)** — runs the same NDA-summarization prompt through each,
and records:

- **Telemetry / latency** per pipeline stage (pre-processing, inference, post-processing) plus peak GPU memory.
- **Accuracy scores** for each precision mode, using semantic similarity against NDA concept gold standards.
- **Demo outputs** — the actual generated summaries — so you can read what the model produced at every precision.

The result is a clean, reproducible comparison of the classic quantization trade-off: how much
speed and VRAM you save versus how much answer quality you give up.

---

## Project Overview

Large language models like SaulLM-7B are powerful but memory-hungry — a full FP16 load of a 7B model
needs roughly 14 GB of VRAM, which does not comfortably fit on a free-tier T4 GPU once activations and
the KV cache are included. **Quantization** shrinks the model by storing weights in fewer bits, trading
a small amount of numerical precision for large savings in memory and (often) speed.

This project answers a practical question for anyone deploying an LLM on constrained hardware:

> *If I quantize SaulLM-7B from 16-bit down to 8-bit or 4-bit, how much do I save — and does the legal
> summary still capture the clauses that matter?*

To make the answer concrete and measurable, the harness:

1. Loads a real mock NDA and wraps it in a Mistral-style instruction prompt asking for exactly three
   bullet points: **Confidential Information**, **Receiving Party Obligations**, and **Governing Law**.
2. Runs the same prompt at each precision while timing every stage and tracking peak VRAM.
3. Scores each generated summary against gold-standard descriptions of the three NDA concepts using
   sentence-embedding cosine similarity.
4. Writes machine-readable logs (CSV/JSON) and a human-readable transcript for presentations.

### Why SaulLM and NDAs?

`Equall/Saul-7B-Instruct-v1` is an instruction-tuned model specialized for the **legal domain**, built
on a Mistral-7B backbone. Pairing it with an NDA-summarization task makes the quality evaluation
domain-relevant: a good summary is not just fluent, it correctly identifies confidentiality scope,
the receiving party's duties, and the governing jurisdiction.

---

## My Contribution

I designed and built the entire benchmarking system around the off-the-shelf SaulLM-7B weights.
Concretely, my work in this repository covers:

- **Quantization engine** (`src/engine/model_loader.py`) — configurable loaders for FP16, 8-bit
  (LLM.int8()), and 4-bit (NF4 with double quantization) using Hugging Face Transformers +
  bitsandbytes, including device-map placement and CPU/disk offload safeguards so the 16-bit baseline
  can still load on a 15 GB T4.
- **Benchmark orchestrator** (`scripts/run_benchmark.py`) — a CLI that runs every precision in a fixed
  `4-bit → 8-bit → 16-bit` order, wipes GPU memory between runs, handles out-of-memory failures
  gracefully (with an automatic stronger-offload retry for FP16), and emits all logs and transcripts.
- **Telemetry layer** (`src/telemetry/metrics.py`) — a `PerformanceTracker` that times each pipeline
  stage with `perf_counter` and captures peak VRAM via CUDA memory stats.
- **Semantic accuracy metric** (`src/evaluation/accuracy.py`) — a rubric-based scorer that embeds the
  model output and three gold-standard NDA concept statements with Sentence-Transformers and grades
  coverage by cosine similarity, mapped to a `[0, 1]` score per concept.
- **Domain prompt pipeline** (`src/data/prompt_pipeline.py`) — reads the raw NDA and builds the
  Mistral-compliant `[INST] … [/INST]` instruction prompt.
- **Model profiler** (`src/telemetry/model_profiler.py`) — a slide-ready report of parameter counts,
  estimated VRAM by precision, layer-type counts, and component-level (attention / FFN / embedding /
  norm) parameter distribution.
- **Demo notebook** (`notebooks/quantization_demo.ipynb`) — a guided walkthrough that installs deps,
  verifies the GPU, runs the benchmark, and renders the latency/accuracy tables and charts.

The model weights themselves are not mine — they are the publicly released `Equall/Saul-7B-Instruct-v1`.
Everything that loads, quantizes, benchmarks, scores, and reports on them is original work.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep learning core | [PyTorch](https://pytorch.org/) (`torch>=2.0.0`) |
| Model & tokenizer | [Hugging Face Transformers](https://huggingface.co/docs/transformers) (`>=4.35.0`) |
| Device placement / offload | [Accelerate](https://huggingface.co/docs/accelerate) (`>=0.24.1`) |
| Quantization | [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) (`>=0.41.1`) — LLM.int8() & NF4 |
| Semantic evaluation | [Sentence-Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) + scikit-learn cosine similarity |
| Analysis & charting | pandas, matplotlib |
| Target model | `Equall/Saul-7B-Instruct-v1` (legal, Mistral-7B backbone) |
| Runtime | Google Colab (T4/A100 GPU) or local CUDA GPU |

---

## Repository Structure

```
SaulLM-Quantization-Project/
├── README.md
├── requirements.txt              # Pinned dependency set
├── configs/
│   └── experiment_config.yaml    # Placeholder for experiment configuration
├── scripts/
│   └── run_benchmark.py          # Main CLI: runs all precisions, writes logs + transcripts
├── notebooks/
│   └── quantization_demo.ipynb   # Guided Colab demo (install → GPU check → run → charts)
├── outputs/                      # Generated artifacts (logs + demo responses)
│   ├── metrics_log.csv           #   stage latency + peak VRAM per precision
│   ├── accuracy_log.csv          #   semantic accuracy per precision (created on run)
│   ├── demo_responses.txt        #   human-readable transcript of generated summaries
│   └── demo_responses.json       #   machine-readable transcript (created on run)
└── src/
    ├── data/
    │   ├── prompt_pipeline.py     # Builds the Mistral-style [INST] NDA prompt
    │   └── raw_documents/
    │       └── mock_nda.txt       # The mock NDA used as benchmark input
    ├── engine/
    │   └── model_loader.py        # FP16 / 8-bit / 4-bit loaders + offload safeguards
    ├── evaluation/
    │   └── accuracy.py            # Sentence-embedding semantic accuracy scorer
    └── telemetry/
        ├── metrics.py             # PerformanceTracker (per-stage latency + peak VRAM)
        └── model_profiler.py      # Parameter / VRAM / architecture profiling report
```

---

## Features

- **Three-way precision benchmark** — FP16, 8-bit LLM.int8(), and 4-bit NF4 in one run, always in a
  memory-safe `4-bit → 8-bit → 16-bit` order.
- **Per-stage telemetry** — separate latency and peak-VRAM measurements for pre-processing, inference,
  and post-processing, so you can see exactly where time is spent.
- **Semantic accuracy scoring** — output quality is graded by meaning, not keyword matching, against
  gold-standard statements for the three core NDA concepts.
- **OOM resilience** — quantized loads stay GPU-first; the FP16 baseline uses CPU/disk offload and an
  automatic stronger-offload retry, so a single OOM does not abort the whole sweep.
- **Configurable memory caps** — CLI flags control GPU/CPU memory budgets, input truncation, and
  generation length to fit different GPUs (tuned defaults for Colab T4).
- **Presentation-ready outputs** — CSV for analysis, JSON for tooling, and a formatted text transcript
  of the actual model responses.
- **Model profiler** — slide-ready breakdown of parameters, estimated VRAM per precision, layer types,
  and attention/FFN/embedding/norm parameter distribution.

---

## Environment Setup

### A. Google Colab (recommended)

#### A0. Fresh Colab bootstrap (run exactly as-is)
Run these cells first in Colab to ensure a clean clone of the repo:
```bash
# Clean up any previous failed clones
!rm -rf saullm-quantization-project

# Clone the repository
!git clone https://github.com/moosayousaf/saullm-quantization-project.git

# Move into the correctly named lowercase folder
%cd saullm-quantization-project

# Step 1: install dependencies
!pip install -q -r requirements.txt
```

1. Open Colab and set runtime to **GPU** (`Runtime -> Change runtime type -> T4/A100`).
2. Clone your repository and enter it:
   ```bash
   !git clone <your-repo-url>
   %cd SaulLM-Quantization-Project
   ```
3. Install dependencies:
   ```bash
   !pip install -r requirements.txt
   ```

### B. Local Python setup
1. Python 3.10+ recommended.
2. Create environment and install:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Ensure CUDA-compatible PyTorch and GPU drivers are correctly installed.

### Optional: Hugging Face authentication
The model is publicly available, but if you hit a gated/rate-limited download you can authenticate by
exporting a token before running — the loader reads `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`:
```bash
export HF_TOKEN=hf_your_token_here
```

---

## Running the Benchmark

From the repository root:
```bash
# Full run (all precisions, always ordered 4-bit -> 8-bit -> 16-bit)
python scripts/run_benchmark.py --max-cpu-memory 64GiB --fp16-gpu-memory 8GiB --fp16-retry-gpu-memory 6GiB

# Colab T4 run including 16-bit baseline with stronger FP16 offload safeguards
python scripts/run_benchmark.py --precisions 4-bit,8-bit,16-bit --max-new-tokens 256 --max-input-tokens 1536 --max-gpu-memory 12GiB --max-cpu-memory 64GiB --fp16-gpu-memory 8GiB --fp16-retry-gpu-memory 6GiB

# Optional quick demo run (4-bit only)
python scripts/run_benchmark.py --precisions 4-bit --max-new-tokens 256 --max-input-tokens 2048 --max-gpu-memory 12GiB

# Optional 16-bit baseline attempt with stronger offload
python scripts/run_benchmark.py --precisions 16-bit --max-new-tokens 256 --max-input-tokens 1536 --max-gpu-memory 12GiB --max-cpu-memory 64GiB --fp16-gpu-memory 8GiB --fp16-retry-gpu-memory 6GiB
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--model-id` | `Equall/Saul-7B-Instruct-v1` | Hugging Face model id to benchmark. |
| `--precisions` | `4-bit,8-bit,16-bit` | Comma-separated precisions (aliases `fp16`/`baseline` accepted). Run order is always `4-bit → 8-bit → 16-bit`. |
| `--max-new-tokens` | `256` | Max generated tokens per run. |
| `--max-input-tokens` | `2048` | Tokenizer truncation cap for the input prompt. |
| `--max-gpu-memory` | `12GiB` | GPU memory cap for weight placement. |
| `--max-cpu-memory` | `48GiB` | CPU RAM cap for offloading. |
| `--fp16-gpu-memory` | `8GiB` | GPU cap specifically for the 16-bit baseline (forces safer offload). |
| `--fp16-retry-gpu-memory` | `6GiB` | GPU cap for the 16-bit retry if the first load OOMs. |
| `--offload-folder` | `offload` | Folder for CPU/disk offloaded weights. |

### Generated artifacts
- `outputs/metrics_log.csv` → stage latency + peak VRAM by precision.
- `outputs/accuracy_log.csv` → NDA concept coverage accuracy by precision.
- `outputs/demo_responses.txt` and `outputs/demo_responses.json` → generated outputs for presentation.

### If Colab crashes or hangs on load
1. Restart runtime and run only one benchmark command per session.
2. Verify free GPU memory before starting: `!nvidia-smi`.
3. Start with 4-bit only: `python scripts/run_benchmark.py --precisions 4-bit --max-new-tokens 256 --max-input-tokens 1536 --max-gpu-memory 12GiB`.
4. Then run 8-bit separately. Run the 16-bit baseline last (or skip it on T4 if unstable).

---

## Demo Flow

Use `notebooks/quantization_demo.ipynb`.

Notebook flow:
1. install deps,
2. verify GPU,
3. run the benchmark script,
4. display latency and accuracy tables,
5. plot the latency chart,
6. print model responses for each precision,
7. architecture profiler output (for model-structure / parameter reporting).

### Accuracy log columns explained
- `Accuracy`: overall score = mean of the three concept scores.
- `ConfidentialInformationScore`, `ObligationsScore`, `GoverningLawScore`: per-concept coverage scores in `[0, 1]`.
- `*MatchedKeywords`: compatibility field representing concept semantic similarity as a percentage (0–100).
- `*TotalKeywords`: compatibility denominator fixed at `100` for the semantic percentage scale.
- `Error`: populated only when a run fails (`oom` or `error` status).

---

## How the Project Works

### Model and quantization
- Model: `Equall/Saul-7B-Instruct-v1` (causal LM).
- Precision modes:
  - **16-bit**: FP16 (alias: `baseline` / `fp16`).
  - **8-bit**: LLM.int8() via bitsandbytes (`llm_int8_threshold=6.0`).
  - **4-bit**: NF4 quantization via bitsandbytes with double quantization and FP16 compute dtype.
- Quantized loads are placed GPU-first (`device_map={"": 0}`) to avoid heavy CPU RAM pressure; the FP16
  baseline uses sequential placement with CPU/disk offload.

### Pipeline stages
Each benchmark run measures three stages:
1. **Pre-processing**: tokenize prompt.
2. **Inference**: `model.generate(...)` forward pass.
3. **Post-processing**: decode generated tokens.

VRAM is reset and peak-measured per stage via `torch.cuda.reset_peak_memory_stats()` /
`max_memory_allocated()`, and timing uses `time.perf_counter()`.

### NDA prompt formatting
`src/data/prompt_pipeline.py` reads `src/data/raw_documents/mock_nda.txt` and wraps the instruction +
document into a Mistral-style `[INST] … [/INST]` prompt that asks for exactly three labeled bullets.

### Accuracy metric
`src/evaluation/accuracy.py` implements a semantic-similarity score (Sentence-Transformers,
`all-MiniLM-L6-v2`) for three required NDA concepts:
1. Confidential information definition,
2. Receiving-party obligations,
3. Governing law.

Each concept receives a cosine-similarity-based score in `[0, 1]` (the raw `[-1, 1]` similarity is
linearly mapped to `[0, 1]`); the final accuracy is the average over the three concept scores.

---

## Model Structure, Parameter Count, and Compute-Cost Discussion

The profiler in `src/telemetry/model_profiler.py` (`profile_model(model)`) inspects:
- total / trainable / frozen parameter counts,
- estimated VRAM usage by precision (FP16 ≈ 2 B, 8-bit ≈ 1 B, 4-bit ≈ 0.5 B bytes per parameter),
- layer-type counts and key config values (hidden size, heads, layers, vocab, …),
- component-level parameter distribution (embedding / attention / feed-forward / layer norm).

This makes it easy to connect the measured latency and VRAM numbers back to the model's architecture
and to explain *why* 4-bit fits where 16-bit does not.

---

## License & Attribution

The benchmarking code in this repository is original work. The benchmarked model,
`Equall/Saul-7B-Instruct-v1`, is released by its respective authors under its own license — review and
comply with the model card terms on Hugging Face before redistribution or commercial use.
