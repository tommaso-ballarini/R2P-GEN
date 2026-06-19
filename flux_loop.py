"""
R2P-GEN Full Pipeline Orchestrator (FLUX Edition)

Questo modulo orchestra la pipeline R2P-GEN modulare:
- Stage 1: generate_only (Estrazione + FLUX Img2Img)
- Stage 2: verify_base (Verifica con Qwen3-VL + CLIP)
- Stage 3: refine (Aggancio API per rigenerare i falliti)
- Stage 4: final_judge (Valutazione severa con InternVL2)
- Stage 5: full_auto (Esegue tutto in sequenza)
- Stage 6: text_fix (Nuovo stage per correzioni testuali)
"""

import os
import json
import torch
import argparse
import numpy as np

from pipeline.generate import Generator
from pipeline.verify import verify_generation_r2p, _extract_attributes_for_clip
from pipeline.judge import FinalJudge
from pipeline.r2p_tools import ClipScoreCalculator
from pipeline.utils2 import cleanup_gpu, ensure_output_dir
from r2p_core.models.qwen3_vl_reasoning import Qwen3VLReasoning  # Step 1 output
from config import Config
import openai
from pipeline.refine import qwen3_rewrite_prompt, generate_recovery_http, _generate_batch_http
from pipeline.prompts.flux_prompts import build_flux_prompt


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_first_image(content: dict) -> str | None:
    """
    Restituisce la representative_image se presente,
    altrimenti il primo elemento di top_k_images,
    altrimenti il primo di image/selected_images.
    """
    if content.get("representative_image"):
        return content["representative_image"]
    
    top_k = content.get("top_k_images")
    if top_k:
        return top_k[0]

    value = content.get("image") or content.get("selected_images")
    if isinstance(value, list):
        return value[0] if value else None
    return value



def _get_best_reference_for_recovery(
    content: dict,
    missing_details: list,
    clip_calculator,
) -> str:
    """
    Tra le top_k_images, seleziona quella con CLIP score più alto
    rispetto agli attributi mancanti. Fallback su representative_image.
    """
    from PIL import Image

    candidates = content.get("top_k_images") or []
    if not candidates:
        return _get_first_image(content)

    if not missing_details:
        return candidates[0]

    best_path, best_score = candidates[0], -1.0
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            img = Image.open(path).convert("RGB")
            score, _ = clip_calculator.compute_attribute_score(img, missing_details)
            if score > best_score:
                best_score = score
                best_path = path
        except Exception:
            continue

    return best_path


def _build_reasoner() -> Qwen3VLReasoning:
    """Istanzia Qwen3VLReasoning con i parametri da Config."""
    return Qwen3VLReasoning(
        model_path=Config.Models.QWEN3_MODEL,  # ← Qwen3-VL path
        device="cuda",
        torch_dtype=torch.bfloat16,            # Qwen3-VL è addestrato in bfloat16, non float16
        attn_implementation="flash_attention_2", # Qwen3 supporta FA2, più efficiente di sdpa
        seed=Config.Generate.SEED,
    )


# ---------------------------------------------------------------------------
# Stage 1 — Generate Only
# ---------------------------------------------------------------------------

def stage_generate_only(
    database_path: str,
    output_dir: str,
    num_shards: int = 1,
    shard_index: int = 0,
) -> dict:
    """Fase 1: Estrazione + FLUX Img2Img (con sharding opzionale)."""
    print(f"\n{'='*70}\n🚀 STAGE: GENERATE ONLY "
          f"[shard {shard_index}/{num_shards}]\n{'='*70}")
    ensure_output_dir(output_dir)

    generator = Generator(
        database_path=database_path,
        output_dir=output_dir,
        num_shards=num_shards,      # Generator deve accettare questi kwargs;
        shard_index=shard_index,    # se non li accetta ancora, aggiungili lì.
    )
    stats = generator.generate_all()
    generator.cleanup()
    cleanup_gpu()

    print(f"\n📊 Generazione: {stats['success']} OK, {stats['failed']} falliti.")
    return stats


# ---------------------------------------------------------------------------
# Stage 2 — Verify Base
# ---------------------------------------------------------------------------

def stage_verify_base(database_path: str, output_dir: str) -> str:
    """Fase 2: Verifica con Qwen3-VL + CLIP. Salva rejected_concepts.json."""
    print(f"\n{'='*70}\n📍 STAGE: VERIFY BASE (Qwen3-VL)\n{'='*70}")

    with open(database_path, "r", encoding="utf-8") as f:
        database = json.load(f)
    concept_dict = database.get("concept_dict", {})

    print("   Caricamento modelli di verifica...")
    reasoner = _build_reasoner()
    clip_calculator = ClipScoreCalculator(device="cuda")

    verified_count = 0
    rejected_dict = {}

    for concept_id, content in concept_dict.items():
        gen_image_path = os.path.join(output_dir, f"{concept_id}_generated.png")

        if not os.path.exists(gen_image_path):
            print(f"   ⚠️  {concept_id}: immagine non trovata → fallita.")
            rejected_dict[concept_id] = {"reason": "Image not found"}
            continue

        ref_image_path = _get_first_image(content)
        if ref_image_path is None:
            print(f"   ⚠️  {concept_id}: nessuna immagine di riferimento → skip.")
            rejected_dict[concept_id] = {"reason": "No reference image"}
            continue

        fingerprints = content.get("info", {})

        verification = verify_generation_r2p(
            reasoner=reasoner,
            clip_calculator=clip_calculator,
            gen_image_path=gen_image_path,
            ref_image_path=ref_image_path,
            fingerprints=fingerprints,
        )

        if verification["is_verified"]:
            verified_count += 1
            print(f"   ✅ {concept_id}: PASS ({verification['score']:.2f})")
        else:
            print(f"   ❌ {concept_id}: FAIL ({verification['score']:.2f})")
            rejected_dict[concept_id] = {
                "score": verification["score"],
                "error_type": "attribute",
                "missing_details": verification.get("failed_attributes", []),
                "details": verification,
            }

    del reasoner
    del clip_calculator
    cleanup_gpu()

    rejected_path = os.path.join(output_dir, "rejected_concepts.json")
    with open(rejected_path, "w", encoding="utf-8") as f:
        json.dump(rejected_dict, f, indent=4)

    total = len(concept_dict)
    print(f"\n📊 Verifica: {verified_count}/{total} passed. "
          f"{len(rejected_dict)} rejected → {rejected_path}")
    return rejected_path


# ---------------------------------------------------------------------------
# Stage 3 — Refine (versione batch)
# ---------------------------------------------------------------------------

def stage_refine(
    database_path: str,
    rejected_path: str,
    output_dir: str,
) -> None:
    """Fase 3: Refine tramite API (closed-loop ospedaliero) in batch su FLUX."""
    print(f"\n{'='*70}\n🚑 STAGE: RECOVERY (BATCH MODE)\n{'='*70}")

    if not os.path.exists(rejected_path):
        print("   ✅ Nessun file rejected trovato. Niente da recuperare!")
        return

    if os.path.getsize(rejected_path) == 0:
        print("   ❌ rejected_concepts.json esiste ma è vuoto (file corrotto?). Abort.")
        return

    with open(rejected_path, "r", encoding="utf-8") as f:
        try:
            rejected_dict = json.load(f)
        except json.JSONDecodeError as e:
            print(f"   ❌ rejected_concepts.json non è JSON valido: {e}. Abort.")
            return

    if not rejected_dict:
        print("   ✅ Lista rejected vuota. Niente da recuperare!")
        return

    if not os.path.exists(database_path):
        print(f"   ❌ Database non trovato: {database_path}. Abort.")
        return

    with open(database_path, "r", encoding="utf-8") as f:
        database = json.load(f)
    concept_dict = database.get("concept_dict", {})

    print(f"   Trovati {len(rejected_dict)} concetti da curare.")

    FLUX_URL       = getattr(Config.Models, "RECOVERY_FLUX_URL", "http://127.0.0.1:8766")
    MAX_ATTEMPTS   = Config.Refine.MAX_ITERATIONS

    print("   Caricamento Qwen3-VL e CLIP in VRAM (GPU 0) per il loop di recovery...")
    reasoner = _build_reasoner()
    clip_calculator = ClipScoreCalculator(device="cuda")

    recovered_count = 0
    graveyard_count = 0
    recovery_report = {}

    # --------------------------------------------------------------------
    # 1. Inizializza lo stato per ogni concept
    # --------------------------------------------------------------------
    concept_states = {}
    for concept_id, fail_data in rejected_dict.items():
        content = concept_dict.get(concept_id, {})
        source_image_path = _get_first_image(content)

        if source_image_path is None:
            print(f"      ⚠️  Nessuna immagine sorgente per {concept_id} → graveyard.")
            recovery_report[concept_id] = {
                "status": "unrecoverable",
                "attempts": 0,
                "final_score": fail_data.get("score", 0.0),
                "last_missing_details": fail_data.get("missing_details", []),
                "original_fail_reason": fail_data.get("missing_details", []),
                "recovered_image_path": None,
                "original_prompt": None,
                "last_rewritten_prompt": None,
                "reason": "No source image",
                "attempts_log": [],
            }
            graveyard_count += 1
            continue

        fingerprints   = content.get("info", {})
        target_context = Config.get_background_template()
        current_prompt = build_flux_prompt(fingerprints, target_context)
        missing_details = fail_data.get("missing_details", [])

        concept_states[concept_id] = {
            "content": content,
            "fingerprints": fingerprints,
            "current_prompt": current_prompt,
            "missing_details": missing_details,
            "is_fixed": False,
            "verification": None,
            "gen_image_path": os.path.join(output_dir, f"{concept_id}_generated.png"),
            "original_prompt": current_prompt,
            "last_rewritten_prompt": None,
            "attempts_history": [],
            "attempts_log": [],
            "original_fail_reason": fail_data.get("missing_details", []),
        }

    # --------------------------------------------------------------------
    # 2. Loop per tentativo – batch su tutti i concept attivi
    # --------------------------------------------------------------------
    for attempt in range(1, MAX_ATTEMPTS + 1):
        active = {cid: s for cid, s in concept_states.items() if not s["is_fixed"]}
        if not active:
            break

        print(f"\n   {'='*50}")
        print(f"   🔄 TENTATIVO {attempt}/{MAX_ATTEMPTS} — {len(active)} concept attivi")
        print(f"   {'='*50}")

        # ---- 2a. Rewrite prompt per tutti (sequenziale su Qwen3) ----
        prompts_batch, seeds_batch, paths_batch, cids_batch, sources_batch = [], [], [], [], []

        for concept_id, state in active.items():
            missing_details_before = list(state["missing_details"])

            best_source = _get_best_reference_for_recovery(
                state["content"], state["missing_details"], clip_calculator
            )

            new_prompt = qwen3_rewrite_prompt(
                reasoner=reasoner,
                original_prompt=state["current_prompt"],
                missing_details=state["missing_details"],
                attempt=attempt,
                failed_image_path=state["attempts_log"][-1].get("image_path") if state["attempts_log"] else None,
                attempts_history=state["attempts_history"],
            )
            print(f"      [{concept_id}] 📝 {new_prompt[:70]}...")

            attempt_image_path = os.path.join(
                output_dir, f"{concept_id}_generated_attempt{attempt}.png"
            )
            new_seed = Config.Generate.SEED + attempt

            prompts_batch.append(new_prompt)
            seeds_batch.append(new_seed)
            paths_batch.append(attempt_image_path)
            cids_batch.append(concept_id)
            sources_batch.append(best_source)

            # Salva nel contesto per dopo
            state["last_rewritten_prompt"] = new_prompt
            state["_missing_details_before"] = missing_details_before
            state["_best_source"] = best_source

        # ---- 2b. Batch FLUX (una chiamata HTTP per tutti) ----
        print(f"\n      🚀 Invio batch FLUX: {len(prompts_batch)} immagini...")
        batch_results = _generate_batch_http(
            flux_url=FLUX_URL,
            source_image_paths=sources_batch,
            prompts=prompts_batch,
            seeds=seeds_batch,
            output_paths=paths_batch,
        )

        # ---- 2c. Verifica per ogni concept (sequenziale) ----
        for i, concept_id in enumerate(cids_batch):
            state = concept_states[concept_id]
            attempt_image_path = paths_batch[i]
            success = batch_results[i]
            new_prompt = state["last_rewritten_prompt"]
            missing_details_before = state["_missing_details_before"]

            if not success:
                print(f"      [{concept_id}] ⚠️ Generazione FLUX fallita.")
                state["attempts_log"].append({
                    "attempt": attempt,
                    "prompt_used": new_prompt,
                    "clip_score": None,
                    "vlm_avg": None,
                    "missing_attributes": state["missing_details"],
                    "success": False,
                    "image_path": None,
                })
                continue

            verification = verify_generation_r2p(
                reasoner=reasoner,
                clip_calculator=clip_calculator,
                gen_image_path=attempt_image_path,
                ref_image_path=state["_best_source"],
                fingerprints=state["fingerprints"],
            )

            # Estrai punteggi aggregati
            clip_scores_list = list(verification["clip_details"]["gen"].values()) if "clip_details" in verification else []
            clip_score = float(np.mean(clip_scores_list)) if clip_scores_list else None
            vlm_scores = [item["score"] for item in verification.get("vlm_history", []) if item.get("phase") == "single_check"]
            vlm_avg = float(np.mean(vlm_scores)) if vlm_scores else None

            state["attempts_log"].append({
                "attempt": attempt,
                "prompt_used": new_prompt,
                "clip_score": clip_score,
                "vlm_avg": vlm_avg,
                "missing_attributes": verification.get("failed_attributes", state["missing_details"]),
                "success": verification["is_verified"],
                "image_path": attempt_image_path,
            })

            state["attempts_history"].append({
                "attempt": attempt,
                "missing_before": missing_details_before,
                "missing_after": verification.get("failed_attributes", state["missing_details"]),
                "improved": len(verification.get("failed_attributes", state["missing_details"])) < len(missing_details_before),
            })

            state["verification"] = verification

            if verification["is_verified"]:
                print(f"      [{concept_id}] ✅ GUARITO al tentativo {attempt}!")
                import shutil
                # Rinomina la precedente _generated.png come rejected prima di sovrascriverla
                previous_attempt_num = attempt - 1
                rejected_name = os.path.join(
                    output_dir,
                    f"{concept_id}_generated_rejected_attempt{previous_attempt_num}.png"
                )
                if os.path.exists(state["gen_image_path"]):
                    os.rename(state["gen_image_path"], rejected_name)
                    print(f"         📦 Precedente salvata come: {os.path.basename(rejected_name)}")

                shutil.copy2(attempt_image_path, state["gen_image_path"])
                state["is_fixed"] = True
                recovered_count += 1
            else:
                state["missing_details"] = verification.get("failed_attributes", state["missing_details"])
                MAX_ATTR_LENGTH = 80  # attributi più lunghi sono quasi certamente allucinazioni narrative
                state["missing_details"] = [
                    a for a in state["missing_details"]
                    if len(a) <= MAX_ATTR_LENGTH
                ]
                # Se il filtro ha eliminato tutto, torna ai missing originali troncati
                if not state["missing_details"]:
                    state["missing_details"] = [
                        a for a in state["original_fail_reason"]
                        if len(a) <= MAX_ATTR_LENGTH
                    ][:3]
                state["current_prompt"] = new_prompt
                print(f"      [{concept_id}] ❌ Permangono: {state['missing_details']}")

    # --------------------------------------------------------------------
    # 3. Compilazione report finale
    # --------------------------------------------------------------------
    for concept_id, state in concept_states.items():
        is_fixed = state["is_fixed"]
        verification = state["verification"]

        if is_fixed:
            # Trova il tentativo in cui è guarito
            attempt_success = None
            for log in reversed(state["attempts_log"]):
                if log.get("success"):
                    attempt_success = log["attempt"]
                    break
            recovery_report[concept_id] = {
                "status": "recovered",
                "attempts": attempt_success or 0,
                "final_score": verification.get("score", 0.0) if verification else 0.0,
                "last_missing_details": [],
                "recovered_image_path": state["gen_image_path"],
                "original_prompt": state["original_prompt"],
                "last_rewritten_prompt": state["last_rewritten_prompt"],
                "original_fail_reason": state["original_fail_reason"],
                "reason": "",
                "attempts_log": state["attempts_log"],
            }
        else:
            graveyard_count += 1
            recovery_report[concept_id] = {
                "status": "unrecoverable",
                "attempts": MAX_ATTEMPTS,
                "final_score": verification.get("score", 0.0) if verification else 0.0,
                "last_missing_details": state["missing_details"],
                "recovered_image_path": None,
                "original_prompt": state["original_prompt"],
                "last_rewritten_prompt": state["last_rewritten_prompt"],
                "original_fail_reason": state["original_fail_reason"],
                "reason": "Max attempts reached",
                "attempts_log": state["attempts_log"],
            }

    # Salva report
    report_path = os.path.join(output_dir, "recovery_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(recovery_report, f, indent=4)
    print(f"\n📋 Report recupero salvato → {report_path}")
    print(f"\n📊 Recovery: {recovered_count} salvati | {graveyard_count} graveyard")

    del reasoner
    del clip_calculator
    cleanup_gpu()

# ---------------------------------------------------------------------------
# Stage 4 — Final Judge
# ---------------------------------------------------------------------------

def stage_final_judge(database_path: str, output_dir: str) -> None:
    """Fase 4: Giudizio finale indipendente (InternVL2)."""
    print(f"\n{'='*70}\n⚖️  STAGE: FINAL JUDGE\n{'='*70}")

    with open(database_path, "r", encoding="utf-8") as f:
        database = json.load(f)

    judge = FinalJudge(
        threshold=Config.Refine.TARGET_ACCURACY,
        use_dino=True,
        use_clip=True,
        use_vqa=True,
    )

    concept_dict = database.get("concept_dict", {})
    results = {}

    for concept_id, content in concept_dict.items():
        gen_image_path = os.path.join(output_dir, f"{concept_id}_generated.png")
        if not os.path.exists(gen_image_path):
            continue

        ref_image_path = _get_first_image(content)
        if ref_image_path is None:
            print(f"   ⚠️  {concept_id}: nessuna reference → skip.")
            continue

        fingerprints = content.get("info", {})

        try:
            judge_eval = judge.evaluate(
                generated_image=gen_image_path,
                reference_image=ref_image_path,
                fingerprints=fingerprints,
                prompt=fingerprints.get("flux_prompt",
                       fingerprints.get("sdxl_prompt", "")),
            )
            results[concept_id] = judge_eval.to_dict()
            status = "✅" if judge_eval.passed else "❌"
            print(f"   {status} {concept_id} | "
                  f"TIFA: {judge_eval.tifa_score:.1%} | "
                  f"DINO: {judge_eval.dino_i:.3f}")
        except Exception as e:
            print(f"   ⚠️ Errore su {concept_id}: {e}")

    judge.cleanup()

    results_path = os.path.join(output_dir, "final_judge_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\n📁 Risultati finali → {results_path}")

# ---------------------------------------------------------------------------
# Stage 5 — Text Fix (NUOVO STAGE)
# ---------------------------------------------------------------------------

def stage_text_fix(database_path: str, output_dir: str) -> None:
    """
    Fase 5: Correzione testuale (placeholder).
    Da implementare secondo le esigenze specifiche.
    """
    print(f"\n{'='*70}\n✏️  STAGE: TEXT FIX\n{'='*70}")
    print(f"   Database: {database_path}")
    print(f"   Output dir: {output_dir}")
    print("   [TODO] Implementare la logica di correzione testuale.")
    # Qui puoi aggiungere il codice per la correzione dei testi,
    # ad esempio caricare il database, elaborare i prompt, correggere descrizioni, ecc.

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R2P-GEN Modulare (FLUX Edition)")
    parser.add_argument("--database", type=str, required=True)
    parser.add_argument("--output",   type=str, default="output")
    # MODIFICA: Aggiunto "text_fix" alle scelte dello stage
    parser.add_argument("--stage",
                        choices=["generate_only", "verify_base",
                                 "refine", "text_fix", "final_judge", "full_auto"],
                        required=True)
    parser.add_argument("--num-shards",  type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    rejected_path = os.path.join(args.output, "rejected_concepts.json")

    if args.stage == "generate_only":
        stage_generate_only(args.database, args.output,
                            args.num_shards, args.shard_index)

    elif args.stage == "verify_base":
        stage_verify_base(args.database, args.output)

    elif args.stage == "refine":
        stage_refine(args.database, rejected_path, args.output)

    # MODIFICA: Aggiunto il branch per il nuovo stage
    elif args.stage == "text_fix":
        stage_text_fix(args.database, args.output)

    elif args.stage == "final_judge":
        stage_final_judge(args.database, args.output)

    elif args.stage == "full_auto":
        print("\n🚀 AVVIO PIPELINE FULL AUTO")
        stage_generate_only(args.database, args.output,
                            args.num_shards, args.shard_index)
        rejected_path = stage_verify_base(args.database, args.output)
        stage_refine(args.database, rejected_path, args.output)
        stage_final_judge(args.database, args.output)
        print("\n🏁 PIPELINE COMPLETATA")