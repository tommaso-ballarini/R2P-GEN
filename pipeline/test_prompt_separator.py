"""
test_prompt_separator.py

Test rapido: confronta 4 strategie di separazione prompt su 3 template
di backpack_dog scelti per massimizzare il rischio di bleeding.

Varianti testate:
  flat   → formato attuale, nessuna separazione
  sep    → separazione con punto fermo (vecchia versione)
  scene  → "placed in a scene ..." (separazione con ancoraggio ambientale)
  front  → "photographed in front of ..." (ancoraggio fotografico)

Prerequisito: server FLUX attivo su RECOVERY_FLUX_URL (default 8766).

Output: TEST_OUT_DIR/<concept>_<idx>_<variant>_s<seed>.png
"""

import os
import json
import base64
import requests
from pathlib import Path

FLUX_URL = os.getenv("RECOVERY_FLUX_URL", "http://127.0.0.1:8766")
TEST_OUT = Path("/leonardo_work/IscrC_MUSE/tballari/FM_Data/output/test_separator")
SEEDS    = [42, 1337]

# ---------------------------------------------------------------------------
# Subject phrase per backpack_dog (compilata da compile_subject_phrase)
# ---------------------------------------------------------------------------
SUBJECT_PHRASE = (
    "light gray with blue zipper and pink tongue fabric backpack "
    "with a rectangular with rounded front panel shape, showing dog face design "
    "with embroidered features including ears, eyes, nose, smiling mouth, and tongue"
)

# ---------------------------------------------------------------------------
# 3 template scelti:
#   06 → city background      (controllo neutro, nessun colore in conflitto)
#   08 → blue house           (MASSIMO conflitto: "blue" nel soggetto E nella scena)
#   12 → Eiffel Tower         (nessun conflitto colore, baseline)
# ---------------------------------------------------------------------------
TEST_CASES = {
    "06": "a {0} with a city in the background",
    "08": "a {0} with a blue house in the background",
    "12": "a {0} with the Eiffel Tower in the background",
}


# ---------------------------------------------------------------------------
# Utility: split template in (prefix, scene)
# ---------------------------------------------------------------------------

def _split_template(template: str) -> tuple[str, str]:
    parts  = template.split("{0}", 1)
    prefix = parts[0].strip()          # tipicamente "a"
    scene  = parts[1].strip() if len(parts) > 1 else ""
    return prefix, scene


# ---------------------------------------------------------------------------
# Varianti di build
# ---------------------------------------------------------------------------

def build_flat(template: str, phrase: str) -> str:
    """Formato attuale: nessuna separazione."""
    return template.format(phrase)


def build_sep(template: str, phrase: str) -> str:
    """
    Separazione con punto fermo tra soggetto e scena.
    Es: "a <phrase>. With a blue house in the background."
    """
    prefix, scene = _split_template(template)
    if scene:
        return f"{prefix} {phrase}. {scene[0].upper()}{scene[1:]}."
    return f"{prefix} {phrase}."


def build_scene(template: str, phrase: str) -> str:
    """
    Ancoraggio ambientale esplicito con 'placed in a scene'.
    Es: "a <phrase>, placed in a scene with a blue house in the background"
    Strategia: la virgola chiude la lista attributi; 'placed in a scene'
    segnala a FLUX che ciò che segue è contesto, non proprietà dell'oggetto.
    """
    prefix, scene = _split_template(template)
    if scene:
        return f"{prefix} {phrase}, placed in a scene {scene}"
    return f"{prefix} {phrase}"


def build_front(template: str, phrase: str) -> str:
    """
    Ancoraggio fotografico con 'photographed in front of' / 'against'.
    Funziona bene quando la scena è un landmark o uno sfondo architettonico.
    Es: "a <phrase>, photographed with a blue house in the background"
    Nota: manteniamo 'with ... in the background' dal template originale
    per non alterare il significato, ma aggiungiamo il verbo fotografico
    come separatore semantico forte.
    """
    prefix, scene = _split_template(template)
    if scene:
        return f"{prefix} {phrase}, photographed {scene}"
    return f"{prefix} {phrase}"


# ---------------------------------------------------------------------------
# Tutte le varianti in ordine
# ---------------------------------------------------------------------------

VARIANTS: list[tuple[str, callable]] = [
    ("flat",  build_flat),
    ("sep",   build_sep),
    ("scene", build_scene),
    ("front", build_front),
]


# ---------------------------------------------------------------------------
# Generazione
# ---------------------------------------------------------------------------

def generate_image(prompt: str, seed: int, output_path: Path) -> bool:
    try:
        resp = requests.post(
            f"{FLUX_URL}/generate",
            json={"prompts": [prompt], "seeds": [seed]},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("errors", [""])[0]:
            print(f"   ⚠️  Errore FLUX: {data['errors'][0]}")
            return False

        img_bytes = base64.b64decode(data["images_b64"][0])
        output_path.write_bytes(img_bytes)
        return True

    except Exception as e:
        print(f"   ❌ Errore generazione: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    TEST_OUT.mkdir(parents=True, exist_ok=True)
    print(f"Output → {TEST_OUT}")
    print(f"FLUX   → {FLUX_URL}")

    # Stampa tutti i prompt per ispezione rapida prima di generare
    print("\n" + "=" * 70)
    print("PROMPT COMPARISON (tutte le varianti)")
    print("=" * 70)
    for idx, tmpl in TEST_CASES.items():
        print(f"\n[{idx}] template: {tmpl}")
        for variant_name, fn in VARIANTS:
            print(f"  {variant_name:6s}: {fn(tmpl, SUBJECT_PHRASE)}")
    print()

    # Genera
    results = []
    total   = len(TEST_CASES) * len(VARIANTS) * len(SEEDS)
    done    = 0

    for idx, tmpl in TEST_CASES.items():
        for variant_name, fn in VARIANTS:
            prompt = fn(tmpl, SUBJECT_PHRASE)
            for seed in SEEDS:
                fname = TEST_OUT / f"backpack_dog_{idx}_{variant_name}_s{seed}.png"
                done += 1
                print(f"[{done}/{total}] {fname.name}")
                print(f"         {prompt[:90]}...")
                ok = generate_image(prompt, seed, fname)
                results.append({
                    "idx": idx, "variant": variant_name, "seed": seed,
                    "prompt": prompt, "path": str(fname), "ok": ok,
                })
                print(f"         {'✅' if ok else '❌'}")

    # Salva log
    log_path = TEST_OUT / "test_results.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    ok_count = sum(1 for r in results if r["ok"])
    print(f"\n{'=' * 70}")
    print(f"Completato: {ok_count}/{total} immagini generate")
    print(f"Log → {log_path}")

    # Riepilogo coppie da confrontare
    print(f"\nCoppie da confrontare (per seed):")
    for idx in TEST_CASES:
        for seed in SEEDS:
            print(f"\n  [{idx} s{seed}]")
            for variant_name, _ in VARIANTS:
                print(f"    {variant_name:6s}: backpack_dog_{idx}_{variant_name}_s{seed}.png")


if __name__ == "__main__":
    main()