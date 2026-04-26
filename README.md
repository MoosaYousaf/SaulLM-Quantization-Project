# SaulLM Quantization Project (FP16 vs 8-bit vs 4-bit)

This project benchmarks and demonstrates quantized inference for **`Equall/Saul-7B-Instruct-v1`** on legal NDA text.

It now provides, in one run:
- **Telemetry/latency** per stage: pre-processing, inference, post-processing.
- **Accuracy scores** for each precision mode (`baseline` FP16, `8-bit`, `4-bit`) using a transparent NDA concept rubric.
- **Demo outputs** (actual generated summaries) for professor presentation.

---

## 1) Environment setup

### A. Google Colab (recommended)
### A0. Fresh Colab bootstrap (recommended exactly as-is)
Run these cells first in Colab to ensure a clean clone of your repo:
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

---

## 2) Run the full benchmark + accuracy + demo responses

From repository root:
```bash
# Full run (all precisions)
python scripts/run_benchmark.py

# Colab T4 safer run (recommended first)
python scripts/run_benchmark.py --precisions 4-bit,8-bit --max-new-tokens 96 --max-gpu-memory 12GiB

# Optional FP16 baseline attempt with stronger offload
python scripts/run_benchmark.py --precisions baseline --max-new-tokens 96 --max-gpu-memory 10GiB --max-cpu-memory 64GiB
```

Generated artifacts:
- `outputs/metrics_log.csv` → stage latency + peak VRAM by precision.
- `outputs/accuracy_log.csv` → NDA concept coverage accuracy by precision.
- `outputs/demo_responses.txt` and `outputs/demo_responses.json` → generated outputs for presentation.

---


### Colab memory management notes
- The runner now enables `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation pressure.
- Model loading now uses `max_memory` caps plus CPU/disk offload folder support.
- Quantized loads (4-bit/8-bit) are now GPU-first (no state-dict offload) to avoid Colab CPU RAM spikes during weight materialization.
- If loading appears stuck, restart runtime, rerun from a clean process, and avoid running multiple benchmark processes in one notebook session.
- If a precision still OOMs, the script records status=`oom` in CSV/JSON outputs and continues with remaining precisions instead of crashing.
- For T4 GPUs, run 4-bit/8-bit first and execute baseline FP16 separately.

---


### If Colab still crashes or hangs on load
1. Restart runtime and run only one benchmark command per session.
2. Verify free GPU memory before starting: `!nvidia-smi`.
3. Start with 4-bit only: `python scripts/run_benchmark.py --precisions 4-bit --max-new-tokens 64 --max-gpu-memory 12GiB`.
4. Then run 8-bit separately. Run baseline last (or skip on T4 if unstable).

---

## 3) Notebook demo (for class presentation)

Use:
- `notebooks/quantization_demo.ipynb`

Notebook flow:
1. install deps,
2. verify GPU,
3. run benchmark script,
4. display latency and accuracy tables,
5. plot latency chart,
6. print model responses for each precision,
7. optional architecture profiler output.

---

## 4) How the project works

### Model and quantization
- Model: `Equall/Saul-7B-Instruct-v1` (causal LM).
- Precision modes:
  - **Baseline**: FP16
  - **8-bit**: LLM.int8 via bitsandbytes
  - **4-bit**: NF4 quantization via bitsandbytes

### Pipeline stages
Each benchmark run measures:
1. **Pre-processing**: tokenize prompt.
2. **Inference**: `model.generate(...)` forward pass.
3. **Post-processing**: decode generated tokens.

### NDA prompt formatting
`src/data/prompt_pipeline.py` reads `src/data/raw_documents/mock_nda.txt` and wraps instruction + document into a Mistral-style `[INST] ... [/INST]` prompt.

### Accuracy metric
`src/evaluation/accuracy.py` implements a rubric-based score for three required NDA concepts:
1. Confidential information definition,
2. Receiving-party obligations,
3. Governing law.

Each concept gets 1.0 if detected in output; final accuracy is the average over the 3 concepts.

---

## 5) Model structure, parameter count, and compute-cost discussion

Use the optional profiler in `src/telemetry/model_profiler.py` to inspect:
- total/trainable/frozen parameter counts,
- estimated VRAM usage by precision,
- layer-type counts,
- component-level parameter distribution.

You can run it inside the notebook (included cell) to report:
- model scale,
- memory/computation implications,
- quantization trade-offs.

---

## 6) Performance bottleneck interpretation

In most transformer generation workloads, **inference stage** should dominate latency, while pre/post stages are much smaller. Confirm this from `outputs/metrics_log.csv` and include chart in your slides.

---

## 7) Suggested improvements for your report

You can discuss improvements along these axes:
- **Speed**: quantization, smaller `max_new_tokens`, batching strategies.
- **Memory/model size**: 8-bit/4-bit deployment.
- **Accuracy**: prompt engineering, post-processing guards, better rubric/eval sets.
- **Cost**: choosing precision mode per hardware budget.

---

## 8) Academic and repository requirements checklist

- Every team member should host code in their own GitHub repository/fork.
- If starting from existing public code, include clear attribution/link to original source in your repo.
- Keep this README and notebook as reproducible, step-by-step run instructions.
- Demonstrate with your own data file(s) in addition to the provided mock NDA.

