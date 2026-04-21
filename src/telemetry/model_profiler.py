import torch
from collections import defaultdict

def profile_model(model, print_full_layers=True):
    print("\n" + "="*70)
    print("  SaulLM-7B MODEL PROFILER REPORT")
    print("="*70)
    stats = {}
    stats.update(_param_counts(model))
    stats.update(_memory_estimates(stats["total_params"]))
    stats.update(_architecture_summary(model))
    if print_full_layers:
        _print_layer_table(model)
    _print_component_summary(model)
    print("\n" + "="*70)
    return stats

def _param_counts(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print("\n--- PARAMETER COUNTS ---")
    print(f"  Total parameters   : {total:>15,}")
    print(f"  Trainable params   : {trainable:>15,}")
    print(f"  Frozen params      : {frozen:>15,}")
    print(f"  Total (billions)   : {total/1e9:>15.3f} B")
    return {"total_params": total, "trainable_params": trainable, "frozen_params": frozen}

def _memory_estimates(total_params):
    modes = {"FP16 (baseline)": 2, "8-bit (LLM.int8())": 1, "4-bit (NF4)": 0.5}
    print("\n--- ESTIMATED VRAM USAGE ---")
    memory = {}
    for label, bpp in modes.items():
        gb = total_params * bpp / (1024**3)
        memory[label] = gb
        bar = "#" * int(gb * 2)
        print(f"  {label:<25} : {gb:>5.2f} GB  {bar}")
    print("\n  (T4 GPU has 15 GB VRAM for reference)")
    return {"memory_estimates_gb": memory}

def _architecture_summary(model):
    type_counts = defaultdict(int)
    for module in model.modules():
        type_counts[type(module).__name__] += 1
    sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
    print("\n--- ARCHITECTURE: LAYER TYPE COUNTS ---")
    print(f"  {'Layer Type':<40} {'Count':>6}")
    print(f"  {'-'*40} {'-'*6}")
    for layer_type, count in sorted_types:
        print(f"  {layer_type:<40} {count:>6}")
    config = getattr(model, "config", None)
    if config:
        print("\n--- MODEL CONFIG (key values) ---")
        for key in ["model_type","hidden_size","intermediate_size","num_hidden_layers",
                    "num_attention_heads","num_key_value_heads","max_position_embeddings","vocab_size"]:
            print(f"  {key:<30} : {getattr(config, key, 'N/A')}")
    return {"layer_type_counts": dict(type_counts)}

def _print_layer_table(model):
    print("\n--- PER-LAYER BREAKDOWN ---")
    print(f"  {'Layer Name':<55} {'Type':<25} {'Params':>12}")
    print(f"  {'-'*55} {'-'*25} {'-'*12}")
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue
        params = sum(p.numel() for p in module.parameters())
        display = name if len(name) <= 54 else "..." + name[-51:]
        print(f"  {display:<55} {type(module).__name__:<25} {params:>12,}")

def _print_component_summary(model):
    groups = {"Embedding": 0, "Attention": 0, "Feed-Forward": 0, "Layer Norm": 0, "Other": 0}
    for name, param in model.named_parameters():
        n = name.lower()
        if "embed" in n:
            groups["Embedding"] += param.numel()
        elif any(k in n for k in ["self_attn","q_proj","k_proj","v_proj","o_proj"]):
            groups["Attention"] += param.numel()
        elif any(k in n for k in ["mlp","gate_proj","up_proj","down_proj","feed_forward"]):
            groups["Feed-Forward"] += param.numel()
        elif any(k in n for k in ["norm","ln"]):
            groups["Layer Norm"] += param.numel()
        else:
            groups["Other"] += param.numel()
    total = sum(groups.values())
    print("\n--- COMPONENT SUMMARY (SLIDE-READY) ---")
    print(f"  {'Component':<20} {'Parameters':>14}  {'% of Total':>10}  Bar")
    print(f"  {'-'*20} {'-'*14}  {'-'*10}  {'-'*20}")
    for component, count in groups.items():
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {component:<20} {count:>14,}  {pct:>9.1f}%  {'#' * int(pct/2)}")
