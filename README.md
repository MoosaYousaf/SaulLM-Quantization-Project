# SaulLM Quantization Project (16-bit vs 8-bit vs 4-bit)

This project benchmarks and demonstrates quantized inference for **`Equall/Saul-7B-Instruct-v1`** on legal NDA text.

It now provides, in one run:
- **Telemetry/latency** per stage: pre-processing, inference, post-processing.
- **Accuracy scores** for each precision mode (`16-bit` FP16, `8-bit`, `4-bit`) using semantic similarity against NDA concept gold standards.
- **Demo outputs** actual generated summaries

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
# Full run (all precisions, always ordered 4-bit -> 8-bit -> 16-bit)
python scripts/run_benchmark.py --max-cpu-memory 64GiB --fp16-gpu-memory 8GiB --fp16-retry-gpu-memory 6GiB

# Colab T4 run including 16-bit baseline with stronger FP16 offload safeguards
python scripts/run_benchmark.py --precisions 4-bit,8-bit,16-bit --max-new-tokens 256 --max-input-tokens 1536 --max-gpu-memory 12GiB --max-cpu-memory 64GiB --fp16-gpu-memory 8GiB --fp16-retry-gpu-memory 6GiB

# Optional professor-demo quick run (4-bit only)
python scripts/run_benchmark.py --precisions 4-bit --max-new-tokens 256 --max-input-tokens 2048 --max-gpu-memory 12GiB

# Optional 16-bit baseline attempt with stronger offload
python scripts/run_benchmark.py --precisions 16-bit --max-new-tokens 256 --max-input-tokens 1536 --max-gpu-memory 12GiB --max-cpu-memory 64GiB --fp16-gpu-memory 8GiB --fp16-retry-gpu-memory 6GiB
```

Generated artifacts:
- `outputs/metrics_log.csv` → stage latency + peak VRAM by precision.
- `outputs/accuracy_log.csv` → NDA concept coverage accuracy by precision.
- `outputs/demo_responses.txt` and `outputs/demo_responses.json` → generated outputs for presentation.

---


### If Colab crashes or hangs on load
1. Restart runtime and run only one benchmark command per session.
2. Verify free GPU memory before starting: `!nvidia-smi`.
3. Start with 4-bit only: `python scripts/run_benchmark.py --precisions 4-bit --max-new-tokens 256 --max-input-tokens 1536 --max-gpu-memory 12GiB`.
4. Then run 8-bit separately. Run 16-bit baseline last (or skip on T4 if unstable).

---

## 3) Demo Flow

Use:
- `notebooks/quantization_demo.ipynb`

Notebook flow:
1. install deps,
2. verify GPU,
3. run benchmark script,
4. display latency and accuracy tables,
5. plot latency chart,
6. print model responses for each precision,
7. architecture profiler output (for model-structure/parameter reporting).

### Accuracy log columns explained
- `Accuracy`: overall score = mean of the three concept scores.
- `ConfidentialInformationScore`, `ObligationsScore`, `GoverningLawScore`: per-concept coverage scores in `[0,1]`.
- `*MatchedKeywords`: compatibility field representing concept semantic similarity as a percentage (0-100).
- `*TotalKeywords`: compatibility denominator fixed at `100` for the semantic percentage scale.
- `Error`: populated only when a run fails (`oom` or `error` status).

---

## 4) How the project works

### Model and quantization
- Model: `Equall/Saul-7B-Instruct-v1` (causal LM).
- Precision modes:
  - **16-bit**: FP16 (alias: `baseline` / `fp16`)
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
`src/evaluation/accuracy.py` implements a semantic-similarity score (Sentence-Transformers) for three required NDA concepts:
1. Confidential information definition,
2. Receiving-party obligations,
3. Governing law.

Each concept receives a cosine-similarity-based score in `[0,1]`; final accuracy is the average over the 3 concept scores.

---

## 5) Model structure, parameter count, and compute-cost discussion

Profiler in `src/telemetry/model_profiler.py` to inspect:
- total/trainable/frozen parameter counts,
- estimated VRAM usage by precision,
- layer-type counts,
- component-level parameter distribution.
