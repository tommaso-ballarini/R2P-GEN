# profiler.py
"""
Resource Profiling for R2P-GEN Pipeline

Misura tempo, VRAM, GPU utilization, iterazioni, token count e carbon footprint
per ogni fase della pipeline. Esporta un JSON report adatto a paper scientifici.

Letteratura:
- Tempo/VRAM breakdown: IP-Adapter (Ye et al., 2023), InstantStyle (Wang et al., 2024)
- Carbon footprint: trend NeurIPS 2024-2025 "efficient X"
- Iterazioni a convergenza: Self-Correcting Diffusion (Feng, 2024), TIFA (Hu et al., ICCV 2023)

Usage:
    from pipeline.profiler import ResourceProfiler

    profiler = ResourceProfiler(enable_carbon=True)

    with profiler.phase("SDXL Generation"):
        generate_image(...)

    profiler.record_refinement_iterations(result["iterations"])
    profiler.export("profiling_report.json")
"""

import os
import sys
import json
import time
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, Dict, List, Tuple

# PyTorch (required)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# pynvml (optional - graceful fallback)
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False

# codecarbon (optional - graceful fallback)
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False


class ResourceProfiler:
    """
    Profila risorse computazionali della pipeline R2P-GEN per metriche da paper.

    Metriche misurate:
    - Tempo di esecuzione per fase (time.perf_counter)
    - Peak VRAM allocata per fase (torch.cuda.max_memory_allocated)
    - GPU Utilization % (pynvml, opzionale)
    - Iterazioni a convergenza (tracking manuale)
    - Token count VLM (tracking manuale)
    - Carbon footprint (codecarbon, opzionale)

    Attributes:
        enable_carbon (bool): Se True, attiva tracking CO2 con codecarbon
        phases (dict): Dati per fase {nome_fase: {calls, times[], vram_peaks[]}}
        refinement_iterations (list): Lista di iterazioni per concept
        token_stats (dict): Statistiche token VLM {input_tokens[], output_tokens[]}
        start_time (float): Timestamp inizio profiling
        carbon_tracker (EmissionsTracker): Tracker CO2 (se enable_carbon=True)
    """

    def __init__(self, enable_carbon: bool = False):
        """
        Inizializza il profiler.

        Args:
            enable_carbon: Se True, attiva tracking CO2 (richiede codecarbon)
        """
        self.enable_carbon = enable_carbon
        self.phases = {}  # {phase_name: {"calls": int, "times": [], "vram_peaks": [], "gpu_util": []}}
        self.refinement_iterations = []  # Lista di num iterazioni per concept
        self.token_stats = {"input_tokens": [], "output_tokens": []}
        self.start_time = time.perf_counter()
        self.carbon_tracker = None
        self._current_phase_name = None
        self._current_phase_start = None

        # Inizializza carbon tracker se richiesto
        if self.enable_carbon:
            if not CODECARBON_AVAILABLE:
                print("⚠️  codecarbon not installed. Carbon tracking disabled.")
                print("   Install with: pip install codecarbon")
                self.enable_carbon = False
            else:
                try:
                    self.carbon_tracker = EmissionsTracker(
                        project_name="R2P-GEN",
                        output_dir=".",
                        output_file="emissions_temp.csv",
                        log_level="error"  # Riduce output verboso
                    )
                    self.carbon_tracker.start()
                    print("🌱 Carbon tracking enabled (codecarbon)")
                except Exception as e:
                    print(f"⚠️  Carbon tracker init failed: {e}")
                    self.enable_carbon = False

        # Verifica NVML
        if not NVML_AVAILABLE:
            print("ℹ️  pynvml not available. GPU utilization tracking disabled.")

    @contextmanager
    def phase(self, phase_name: str):
        """
        Context manager per profilare una fase della pipeline.

        Usage:
            with profiler.phase("SDXL Generation"):
                generate_image(...)

        Args:
            phase_name: Nome della fase (es. "SDXL Generation", "VLM Extraction")
        """
        # Inizializza fase se nuova
        if phase_name not in self.phases:
            self.phases[phase_name] = {
                "calls": 0,
                "times": [],
                "vram_peaks": [],
                "gpu_util": []
            }

        # Reset peak VRAM stats per questa fase
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Timestamp inizio
        start_time = time.perf_counter()
        self._current_phase_name = phase_name

        try:
            yield  # Esegui il blocco wrappato
        finally:
            # Timestamp fine
            elapsed = time.perf_counter() - start_time

            # Registra metriche
            phase_data = self.phases[phase_name]
            phase_data["calls"] += 1
            phase_data["times"].append(elapsed)

            # Peak VRAM
            if TORCH_AVAILABLE and torch.cuda.is_available():
                peak_vram_bytes = torch.cuda.max_memory_allocated()
                peak_vram_gb = peak_vram_bytes / (1024 ** 3)
                phase_data["vram_peaks"].append(peak_vram_gb)

            # GPU Utilization
            if NVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    phase_data["gpu_util"].append(util.gpu)  # Percentuale 0-100
                except:
                    pass  # Silent fail se nvml ha problemi

            self._current_phase_name = None

    def record_refinement_iterations(self, iterations: int):
        """
        Registra il numero di iterazioni impiegate per convergere per un concept.

        Args:
            iterations: Numero di iterazioni (da refine.iterative_refinement return dict)
        """
        self.refinement_iterations.append(iterations)

    def record_tokens(self, input_tokens: int, output_tokens: int):
        """
        Registra il token count per una chiamata VLM.

        Args:
            input_tokens: Numero di token in input (immagine + prompt)
            output_tokens: Numero di token generati dal VLM
        """
        self.token_stats["input_tokens"].append(input_tokens)
        self.token_stats["output_tokens"].append(output_tokens)

    def export(self, output_path: str = "profiling_report.json"):
        """
        Esporta il report di profiling in JSON.

        Il JSON è strutturato per essere immediatamente usabile in grafici/tabelle
        di un paper scientifico.

        Args:
            output_path: Path del file JSON di output
        """
        # Stop carbon tracker se attivo
        carbon_data = {}
        if self.enable_carbon and self.carbon_tracker is not None:
            try:
                emissions = self.carbon_tracker.stop()
                carbon_data = {
                    "kwh_consumed": round(float(emissions) if emissions else 0.0, 6),
                    "co2_grams": round(float(emissions * 1000) if emissions else 0.0, 3),
                    "country": "ITA"  # Default - può essere configurato
                }
                # Rimuovi file temporaneo
                if os.path.exists("emissions_temp.csv"):
                    os.remove("emissions_temp.csv")
            except Exception as e:
                print(f"⚠️  Carbon tracker stop failed: {e}")
                carbon_data = {"error": str(e)}

        # Calcola tempo totale
        total_wall_time = time.perf_counter() - self.start_time

        # Aggregazioni per fase
        phases_report = {}
        for phase_name, data in self.phases.items():
            calls = data["calls"]
            times = data["times"]
            vram_peaks = data["vram_peaks"]
            gpu_utils = data["gpu_util"]

            phase_report = {
                "calls": calls,
                "mean_seconds": round(sum(times) / len(times), 2) if times else 0.0,
                "total_seconds": round(sum(times), 2),
            }

            if vram_peaks:
                phase_report["peak_vram_gb"] = round(max(vram_peaks), 2)
                phase_report["mean_vram_gb"] = round(sum(vram_peaks) / len(vram_peaks), 2)

            if gpu_utils:
                phase_report["mean_gpu_utilization_percent"] = round(
                    sum(gpu_utils) / len(gpu_utils), 1
                )

            phases_report[phase_name] = phase_report

        # Refinement stats
        refinement_report = {}
        if self.refinement_iterations:
            refinement_report = {
                "mean_iterations_to_convergence": round(
                    sum(self.refinement_iterations) / len(self.refinement_iterations), 2
                ),
                "max_iterations": max(self.refinement_iterations),
                "min_iterations": min(self.refinement_iterations),
                "total_refine_calls": len(self.refinement_iterations)
            }

        # Token stats
        token_report = {}
        if self.token_stats["input_tokens"]:
            token_report["mean_input_tokens_per_call"] = round(
                sum(self.token_stats["input_tokens"]) / len(self.token_stats["input_tokens"]), 0
            )
            token_report["total_input_tokens"] = sum(self.token_stats["input_tokens"])
        if self.token_stats["output_tokens"]:
            token_report["mean_output_tokens_per_call"] = round(
                sum(self.token_stats["output_tokens"]) / len(self.token_stats["output_tokens"]), 0
            )
            token_report["total_output_tokens"] = sum(self.token_stats["output_tokens"])

        # GPU info
        gpu_name = "Unknown"
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)

        # Build final report
        report = {
            "metadata": {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "gpu": gpu_name,
                "total_wall_time_seconds": round(total_wall_time, 2),
                "torch_available": TORCH_AVAILABLE,
                "nvml_available": NVML_AVAILABLE,
                "codecarbon_available": CODECARBON_AVAILABLE
            },
            "phases": phases_report
        }

        if refinement_report:
            report["refinement"] = refinement_report

        if token_report:
            report["tokens"] = token_report

        if carbon_data:
            report["carbon"] = carbon_data

        # Esporta JSON
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📊 Profiling report exported: {output_path}")
        print(f"   Total time: {total_wall_time:.1f}s")
        print(f"   Phases tracked: {len(phases_report)}")
        if self.refinement_iterations:
            print(f"   Mean iterations/concept: {refinement_report['mean_iterations_to_convergence']}")
        if carbon_data and "kwh_consumed" in carbon_data:
            print(f"   CO2 footprint: {carbon_data['co2_grams']}g CO2eq")


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("\n📊 ResourceProfiler - Test")
    print("─" * 60)

    profiler = ResourceProfiler(enable_carbon=True)

    # Simula fasi della pipeline
    print("\n🔹 Phase 1: VLM Extraction")
    with profiler.phase("VLM Extraction"):
        time.sleep(0.5)  # Simula lavoro
        if TORCH_AVAILABLE and torch.cuda.is_available():
            # Alloca un po' di VRAM per test
            dummy = torch.randn(1000, 1000, device="cuda")
            del dummy

    print("🔹 Phase 2: SDXL Generation")
    with profiler.phase("SDXL Generation"):
        time.sleep(1.0)
        if TORCH_AVAILABLE and torch.cuda.is_available():
            dummy = torch.randn(2000, 2000, device="cuda")
            del dummy

    print("🔹 Phase 3: Verification")
    with profiler.phase("Verification"):
        time.sleep(0.3)

    # Simula refinement iterations
    profiler.record_refinement_iterations(2)
    profiler.record_refinement_iterations(3)
    profiler.record_refinement_iterations(1)

    # Simula token tracking
    profiler.record_tokens(input_tokens=1850, output_tokens=320)
    profiler.record_tokens(input_tokens=1920, output_tokens=285)

    # Esporta
    profiler.export("test_profiling_report.json")

    print("\n✅ Test completato. Controlla test_profiling_report.json")
