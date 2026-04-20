import time
import torch

class PerformanceTracker:
    def __init__(self):
        self.phases = {
            "pre_processing": {"time_sec": 0.0, "peak_memory_mb": 0.0},
            "inference": {"time_sec": 0.0, "peak_memory_mb": 0.0},
            "post_processing": {"time_sec": 0.0, "peak_memory_mb": 0.0}
        }
        self._start_time = None
        self._current_phase = None

    def start_phase(self, phase_name: str):
        if phase_name not in self.phases:
            raise ValueError(f"Unknown phase: {phase_name}. Must be one of {list(self.phases.keys())}")

        self._current_phase = phase_name
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        self._start_time = time.perf_counter()

    def end_phase(self):
        if not self._current_phase:
            raise RuntimeError("Called end_phase() before start_phase()")

        elapsed_time = time.perf_counter() - self._start_time

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak_memory = 0.0

        self.phases[self._current_phase]["time_sec"] = elapsed_time
        self.phases[self._current_phase]["peak_memory_mb"] = peak_memory

        self._current_phase = None
        self._start_time = None

    def print_report(self):
        print("\n" + "="*40)
        print("TELEMETRY REPORT")
        print("="*40)
        total_time = 0.0
        
        for phase, metrics in self.phases.items():
            formatted_phase = phase.replace("_", " ").title()
            print(f"--- {formatted_phase} ---")
            print(f"  Time Elapsed : {metrics['time_sec']:.4f} seconds")
            print(f"  Peak VRAM    : {metrics['peak_memory_mb']:.2f} MB")
            total_time += metrics['time_sec']
            
        print("-" * 40)
        print(f"Total Pipeline Time: {total_time:.4f} seconds")
        print("="*40 + "\n")
