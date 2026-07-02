"""
pipeline/refine_dreambench.py


Refnie for DreamBench benchmark. SEPARATE script from pipeline_orchestrator.stage_refine.
Agreed choice (safer): the refine rewrites ONLY the subject_phrase compiled by Qwen3-VL (e.g., "maroon backpack with colorful patches"), 
NOT the full DreamBench prompt. The official prompt template (e.g., "a {0} in the jungle") 
remains fixed and is reinserted with build_dreambench_prompt() after rewriting. This prevents the recovery 
loop from accidentally altering the context/scene required by the official benchmark, keeping the comparison with DreamBooth cleaner.

"""

import os
import json
import re
import shutil
import argparse

from pipeline.refine import qwen3_rewrite_prompt, _generate_batch_http
from pipeline.verify import verify_generation_r2p
from pipeline.r2p_tools import ClipScoreCalculator
from pipeline.prompts.dreambench_prompts import get_prompts_for_entity_type
from pipeline.prompts.dreambench_prompt_compiler import (
    compile_subject_phrase,
    build_dreambench_prompt,
)
from config import Config


def _build_reasoner():
    import torch
    from r2p_core.models.qwen3_vl_reasoning import Qwen3VLReasoning
    return Qwen3VLReasoning(
        model_path=Config.Models.QWEN3_MODEL,
        device="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        seed=Config.Generate.SEED,
    )


def _sanitize_rewritten_subject_phrase(raw: str) -> str:
    """
    qwen3_rewrite_prompt() use a generic instruction (for complete FLUX prompts 
    with scenery), that explicitly says to "keep the scene/background" and to
    return "a single fluent paragraph". Since here we pass ONLY the subject_phrase
    (without scene), the model sometimes responds with a complete sentence (e.g., "This is a
    maroon backpack with...") instead of a clean noun phrase. This function cleans up the
    most common patterns before reinserting the result into the fixed DreamBench template.
    
    """
    text = raw.strip().strip(".")
    text = re.sub(r"^(this is|it is|it's|that is)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(a|an)\s+", "", text, flags=re.IGNORECASE)
    return text.strip()


def run_dreambench_refine(database_path: str, rejected_path: str, output_dir: str) -> None:
    print(f"\n{'='*70}\n🚑 REFINE DREAMBENCH (subject_phrase only, template fisso)\n{'='*70}")

    if not os.path.exists(rejected_path) or os.path.getsize(rejected_path) == 0:
        print("   ✅ No rejected_dreambench.json or file empty. Nothing to recover!")
        return

    with open(rejected_path, "r", encoding="utf-8") as f:
        rejected_dict = json.load(f)

    if not rejected_dict:
        print("   ✅ List rejected empty. Nothing to recover!")
        return

    with open(database_path, "r", encoding="utf-8") as f:
        database = json.load(f)
    concept_dict = database.get("concept_dict", {})

    FLUX_URL     = getattr(Config.Models, "RECOVERY_FLUX_URL", "http://127.0.0.1:8766")
    MAX_ATTEMPTS = Config.Refine.MAX_ITERATIONS

    print(f"   Found {len(rejected_dict)} images to refine.")
    print("   Loading Qwen3-VL and CLIP in VRAM for the recovery loop...")
    reasoner = _build_reasoner()
    clip_calculator = ClipScoreCalculator(device="cuda")

    # -----------------------------------------------------------------------
    # 1. Initialize state for each rejected image (composite key)
    # -----------------------------------------------------------------------
    states = {}
    for composite_key, fail_data in rejected_dict.items():
        concept_id = fail_data["concept_id"]
        prompt_idx = fail_data["prompt_idx"]
        content    = concept_dict.get(concept_id, {})
        fingerprints = content.get("info", {})
        entity_type  = fingerprints.get("_entity_type", "OBJECT")

        template = get_prompts_for_entity_type(entity_type)[prompt_idx]
        initial_subject_phrase = compile_subject_phrase(fingerprints)

        states[composite_key] = {
            "concept_id":            concept_id,
            "prompt_idx":            prompt_idx,
            "fingerprints":          fingerprints,
            "template":              template,                         
            "current_subject_phrase": initial_subject_phrase,
            "ref_image_path":        fail_data["ref_image_path"],
            "missing_details":       fail_data.get("missing_details", []),
            "original_fail_reason":  fail_data.get("missing_details", []),
            "gen_image_path":        fail_data["gen_image_path"],     
            "is_fixed":              False,
            "verification":          None,
            "attempts_history":      [],
            "attempts_log":          [],
        }

    recovered_count = 0
    graveyard_count = 0

    # -----------------------------------------------------------------------
    # 2. Loop for tentative — batch on all rejected images still active
    # -----------------------------------------------------------------------
    for attempt in range(1, MAX_ATTEMPTS + 1):
        active = {k: s for k, s in states.items() if not s["is_fixed"]}
        if not active:
            break

        print(f"\n   🔄 Tentative {attempt}/{MAX_ATTEMPTS} — {len(active)} active images")

        prompts_batch, seeds_batch, paths_batch, keys_batch, sources_batch = [], [], [], [], []

        for composite_key, state in active.items():
            failed_image_path = (
                state["gen_image_path"] if attempt == 1
                else (state["attempts_log"][-1].get("image_path") if state["attempts_log"] else None)
            )

            raw_rewritten = qwen3_rewrite_prompt(
                reasoner=reasoner,
                original_prompt=state["current_subject_phrase"],
                missing_details=state["missing_details"],
                attempt=attempt,
                failed_image_path=failed_image_path,
                attempts_history=state["attempts_history"],
                fingerprints=state["fingerprints"],
            )
            new_subject_phrase = _sanitize_rewritten_subject_phrase(raw_rewritten)

            new_prompt = build_dreambench_prompt(state["template"], new_subject_phrase)
            print(f"      [{composite_key}] 📝 {new_prompt[:80]}...")

            safe_name = composite_key.replace("<", "").replace(">", "").replace("/", "_")
            attempt_image_path = os.path.join(output_dir, f"{safe_name}_attempt{attempt}.png")
            new_seed = Config.Generate.SEED + attempt * 1000

            prompts_batch.append(new_prompt)
            seeds_batch.append(new_seed)
            paths_batch.append(attempt_image_path)
            keys_batch.append(composite_key)
            sources_batch.append(state["ref_image_path"])

            state["_new_subject_phrase"] = new_subject_phrase
            state["_missing_before"]     = list(state["missing_details"])

        print(f"      🚀 Generation FLUX: {len(prompts_batch)} images (one at a time)...")
        batch_results = []
        for j in range(len(prompts_batch)):
            res = _generate_batch_http(
                flux_url=FLUX_URL,
                source_image_paths=[sources_batch[j]],
                prompts=[prompts_batch[j]],
                seeds=[seeds_batch[j]],
                output_paths=[paths_batch[j]],
            )
            batch_results.append(res[0])

        for i, composite_key in enumerate(keys_batch):
            state = states[composite_key]
            success = batch_results[i]
            attempt_image_path = paths_batch[i]

            if not success:
                print(f"      [{composite_key}] ⚠️ Generation FLUX failed.")
                state["attempts_log"].append({
                    "attempt": attempt, "success": False,
                    "image_path": None, "method": "flux_generation_failed",
                })
                continue

            verification = verify_generation_r2p(
                reasoner=reasoner,
                clip_calculator=clip_calculator,
                gen_image_path=attempt_image_path,
                ref_image_path=state["ref_image_path"],
                fingerprints=state["fingerprints"],
            )

            state["attempts_log"].append({
                "attempt":      attempt,
                "success":      verification["is_verified"],
                "score":        verification.get("score", 0.0),
                "image_path":   attempt_image_path,
                "subject_phrase_used": state["_new_subject_phrase"],
            })
            state["attempts_history"].append({
                "attempt": attempt,
                "missing_before": state["_missing_before"],
                "missing_after":  verification.get("failed_attributes", state["missing_details"]),
                "improved": len(verification.get("failed_attributes", state["missing_details"])) < len(state["_missing_before"]),
            })
            state["verification"] = verification

            if verification["is_verified"]:
                print(f"      [{composite_key}] ✅ Healed at attempt {attempt}!")
                rejected_name = state["gen_image_path"].replace(".png", f"_rejected_attempt{attempt-1}.png")
                if os.path.exists(state["gen_image_path"]):
                    os.rename(state["gen_image_path"], rejected_name)
                shutil.copy2(attempt_image_path, state["gen_image_path"])
                state["is_fixed"] = True
                state["current_subject_phrase"] = state["_new_subject_phrase"]
                recovered_count += 1
            else:
                new_missing = verification.get("failed_attributes", state["missing_details"])
                MAX_ATTR_LENGTH = 80
                new_missing = [a for a in new_missing if len(a) <= MAX_ATTR_LENGTH]
                if not new_missing:
                    new_missing = [a for a in state["original_fail_reason"] if len(a) <= MAX_ATTR_LENGTH][:3]
                state["missing_details"] = new_missing
                state["current_subject_phrase"] = state["_new_subject_phrase"]
                print(f"      [{composite_key}] ❌ Still missing: {state['missing_details']}")

    # -----------------------------------------------------------------------
    # 3. final report
    # -----------------------------------------------------------------------
    recovery_report = {}
    for composite_key, state in states.items():
        if state["is_fixed"]:
            recovery_report[composite_key] = {
                "status": "recovered",
                "final_subject_phrase": state["current_subject_phrase"],
                "recovered_image_path": state["gen_image_path"],
                "attempts_log": state["attempts_log"],
            }
        else:
            graveyard_count += 1
            recovery_report[composite_key] = {
                "status": "unrecoverable",
                "last_missing_details": state["missing_details"],
                "last_subject_phrase": state["current_subject_phrase"],
                "attempts_log": state["attempts_log"],
                "reason": "Max attempts reached",
            }

    report_path = os.path.join(output_dir, "recovery_results_dreambench.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(recovery_report, f, indent=4, ensure_ascii=False)

    print(f"\n📋 Report → {report_path}")
    print(f"📊 Recovery: {recovered_count} healed | {graveyard_count} graveyard")

    del reasoner
    del clip_calculator
    import torch
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refine DreamBench (R2P-GEN)")
    parser.add_argument("--database", type=str, required=True)
    parser.add_argument("--output",   type=str, required=True)
    parser.add_argument("--rejected", type=str, default=None,
                        help="Default: <output>/rejected_dreambench.json")
    args = parser.parse_args()

    rejected_path = args.rejected or os.path.join(args.output, "rejected_dreambench.json")
    run_dreambench_refine(args.database, rejected_path, args.output)