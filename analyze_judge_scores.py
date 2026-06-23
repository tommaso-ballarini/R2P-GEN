#!/usr/bin/env python3
"""
analyze_judge_scores.py

Legge final_judge_results.json e mostra la distribuzione completa dei
final_score, con i percentili principali e un istogramma testuale.

Usalo DOPO aver girato run_judge_test_100.sh per scegliere il threshold
da impostare in Config.Refine.TARGET_ACCURACY.

Usage:
    python analyze_judge_scores.py <path/a/final_judge_results.json>
    python analyze_judge_scores.py /leonardo_work/.../test_100/final_judge_results.json
"""

import json
import sys
import os


def percentile(sorted_vals: list, p: float) -> float:
    """Percentile lineare su lista già ordinata."""
    n = len(sorted_vals)
    idx = p / 100 * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    return sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo])


def histogram(vals: list, bins: int = 20, width: int = 40) -> str:
    """Istogramma testuale normalizzato."""
    lo, hi = min(vals), max(vals)
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in vals:
        b = min(int((v - lo) / step), bins - 1)
        counts[b] += 1
    max_c = max(counts)
    lines = []
    for i, c in enumerate(counts):
        bar_lo = lo + i * step
        bar_hi = bar_lo + step
        bar = "█" * int(c / max_c * width)
        lines.append(f"  {bar_lo:.2f}-{bar_hi:.2f} │{bar:<{width}}│ {c:3d}")
    return "\n".join(lines)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None

    # Auto-detect se non specificato
    if path is None:
        candidates = [
            "/leonardo_work/IscrC_MUSE/tballari/FM_Data/output/test_100/final_judge_results.json",
            "final_judge_results.json",
        ]
        for c in candidates:
            if os.path.exists(c):
                path = c
                break

    if path is None or not os.path.exists(path):
        print("❌ File non trovato. Specifica il path:")
        print("   python analyze_judge_scores.py <path/a/final_judge_results.json>")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    print(f"\n{'='*60}")
    print(f" DISTRIBUZIONE FINAL SCORE — {os.path.basename(path)}")
    print(f" N concetti: {len(data)}")
    print(f"{'='*60}\n")

    scores = sorted(d.get("final_score", 0.0) for d in data.values())
    n = len(scores)

    # ── Statistiche base ──────────────────────────────────────
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / n
    std = variance ** 0.5

    print(f"  Min:    {scores[0]:.3f}   ({scores[0]:.1%})")
    print(f"  Max:    {scores[-1]:.3f}   ({scores[-1]:.1%})")
    print(f"  Media:  {mean:.3f}   ({mean:.1%})")
    print(f"  Std:    {std:.3f}")

    # ── Percentili ───────────────────────────────────────────
    print(f"\n  {'Percentile':<12} {'Score':>8}   {'Pass rate se usato come threshold':>10}")
    print(f"  {'-'*55}")
    for p in [10, 25, 50, 75, 90, 95]:
        v = percentile(scores, p)
        # Se usiamo v come threshold, quanta % dei concetti passa?
        pass_rate = sum(1 for s in scores if s >= v) / n
        print(f"  P{p:02d}         {v:>8.3f}   {pass_rate:>6.1%} dei concetti supererebbe")

    # ── Istogramma ───────────────────────────────────────────
    print(f"\n  Distribuzione (istogramma):\n")
    print(histogram(scores))

    # ── Breakdown per metrica ─────────────────────────────────
    print(f"\n  Breakdown per metrica (media su tutti i concetti):\n")
    metrics = ["clip_i", "clip_t", "dino_i", "tifa_score"]
    for m in metrics:
        vals = [d.get("metrics", {}).get(m, 0.0) for d in data.values()]
        vals = [v for v in vals if v > 0]
        if vals:
            print(f"  {m:<15} media={sum(vals)/len(vals):.3f}  "
                  f"min={min(vals):.3f}  max={max(vals):.3f}  "
                  f"(su {len(vals)}/{n} concetti)")

    # ── Suggerimento threshold ────────────────────────────────
    print(f"\n{'='*60}")
    print(f" SUGGERIMENTO THRESHOLD")
    print(f"{'='*60}")

    p50 = percentile(scores, 50)
    p75 = percentile(scores, 75)
    p90 = percentile(scores, 90)

    print(f"""
  Scegli in base all'obiettivo del progetto:

  ┌─────────────────────────────────────────────────────┐
  │ Obiettivo             │ Threshold │ Pass rate atteso │
  ├─────────────────────────────────────────────────────┤
  │ Permissivo (baseline) │   {p50:.2f}    │     ~50%         │
  │ Bilanciato            │   {p75:.2f}    │     ~25%         │
  │ Aggressivo (alta QA)  │   {p90:.2f}    │     ~10%         │
  └─────────────────────────────────────────────────────┘

  In config.py:
      TARGET_ACCURACY = {p75:.2f}  # bilanciato — modifica a piacere
""")

    # ── Lista concetti falliti ────────────────────────────────
    chosen = p75
    print(f"  Concetti che NON passerebbero con threshold={chosen:.2f}:\n")
    for cid, d in sorted(data.items(), key=lambda x: x[1].get("final_score", 0)):
        s = d.get("final_score", 0)
        if s < chosen:
            m = d.get("metrics", {})
            print(f"  ❌ {cid:<12} score={s:.3f} | "
                  f"CLIP-I={m.get('clip_i',0):.3f} | "
                  f"DINO={m.get('dino_i',0):.3f} | "
                  f"TIFA={m.get('tifa_score',0):.1%}")


if __name__ == "__main__":
    main()