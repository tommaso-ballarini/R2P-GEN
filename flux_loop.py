"""
R2P-GEN Full Pipeline Orchestrator (FLUX Edition)

Questo modulo orchestra la pipeline R2P-GEN modulare:
- Stage 1: generate_only (Estrazione + FLUX Img2Img)
- Stage 2: verify_base (Verifica con MiniCPM + CLIP)
- Stage 3: recovery (Aggancio API per rigenerare i falliti)
- Stage 4: final_judge (Valutazione severa con Qwen2.5-VL)
- Stage 5: full_auto (Esegue tutto in sequenza)
"""

import os
import json
import torch
import gc
import argparse
from pathlib import Path

# Importiamo il nuovo generatore FLUX
from pipeline.generate import Generator
from pipeline.verify import verify_generation_r2p
from pipeline.judge import FinalJudge
from pipeline.r2p_tools import ClipScoreCalculator
from pipeline.utils2 import cleanup_gpu, ensure_output_dir
from r2p_core.models.mini_cpm_reasoning import MiniCPMReasoning
from config import Config

def stage_generate_only(database_path: str, output_dir: str):
    """Fase 1 & 2: Estrazione e Generazione FLUX"""
    print(f"\n{'='*70}\n🚀 STAGE: GENERATE ONLY\n{'='*70}")
    ensure_output_dir(output_dir)
    
    generator = Generator(database_path=database_path, output_dir=output_dir)
    stats = generator.generate_all()
    generator.cleanup()
    cleanup_gpu()
    
    print(f"\n📊 Statistiche Generazione: {stats['success']} completati, {stats['failed']} falliti.")
    return stats

def stage_verify_base(database_path: str, output_dir: str):
    """Fase 3: Verifica Base (MiniCPM). Salva i file rejected per il recovery."""
    print(f"\n{'='*70}\n📍 STAGE: VERIFY BASE (MiniCPM)\n{'='*70}")
    
    with open(database_path, 'r', encoding='utf-8') as f:
        database = json.load(f)
        
    concept_dict = database.get("concept_dict", {})
    
    print("   Caricamento modelli di verifica...")
    reasoner = MiniCPMReasoning(
        model_path=Config.Models.VLM_MODEL,
        device="cuda",
        torch_dtype=torch.float16 if Config.GPU.USE_FP16 else torch.float32,
        attn_implementation="sdpa",
        seed=Config.Generate.SEED
    )
    clip_calculator = ClipScoreCalculator(device="cuda")
    
    verified_count = 0
    rejected_dict = {}
    
    for concept_id, content in concept_dict.items():
        gen_image_path = os.path.join(output_dir, f"{concept_id}_generated.png")
        
        if not os.path.exists(gen_image_path):
            print(f"   ⚠️  {concept_id}: Immagine generata non trovata, la marco come fallita.")
            rejected_dict[concept_id] = {"reason": "Image not found"}
            continue
            
        ref_image_path = content.get("image", content.get("selected_images", []))[0]
        fingerprints = content.get("info", {})
        
        verification = verify_generation_r2p(
            reasoner=reasoner,
            clip_calculator=clip_calculator,
            gen_image_path=gen_image_path,
            ref_image_path=ref_image_path,
            fingerprints=fingerprints
        )
        
        if verification["is_verified"]:
            verified_count += 1
            print(f"   ✅ {concept_id}: PASS ({verification['score']:.2f})")
        else:
            print(f"   ❌ {concept_id}: FAIL ({verification['score']:.2f})")
            
            rejected_dict[concept_id] = {
                "score": verification['score'],
                "error_type": "attribute", 
                "missing_details": verification.get('failed_attributes', []), # Passiamo la lista degli attributi falliti!
                "details": verification
            }

    # Pulizia
    del reasoner
    del clip_calculator
    cleanup_gpu()
    
    # Salvataggio del file ponte per il Recovery
    rejected_path = os.path.join(output_dir, "rejected_concepts.json")
    with open(rejected_path, 'w', encoding='utf-8') as f:
        json.dump(rejected_dict, f, indent=4)
        
    total = len(concept_dict)
    print(f"\n📊 Summary Verifica: {verified_count}/{total} passed. Salvati {len(rejected_dict)} rejected in {rejected_path}")
    return rejected_path

def stage_recovery(rejected_path: str, output_dir: str):
    """Fase 4: Recovery tramite API (Da integrare con i tuoi script)"""
    print(f"\n{'='*70}\n🚑 STAGE: RECOVERY API\n{'='*70}")
    if not os.path.exists(rejected_path):
        print(f"   ✅ Nessun file rejected trovato in {rejected_path}. Niente da recuperare!")
        return
        
    with open(rejected_path, 'r', encoding='utf-8') as f:
        rejected = json.load(f)
        
    if not rejected:
        print("   ✅ Lista rejected vuota. Niente da recuperare!")
        return
        
    print(f"   Trovati {len(rejected)} concetti da recuperare.")
    print("   [!] Qui agganceremo la logica API client-server (5_recovery_pipeline_v2.py)")
    # TODO: Integrare la chiamata allo script API di recovery fornito dall'altra IA

def stage_final_judge(database_path: str, output_dir: str):
    """Fase 5: Giudizio Finale Indipendente (Qwen2.5-VL)"""
    print(f"\n{'='*70}\n⚖️  STAGE: FINAL JUDGE (Qwen2.5-VL)\n{'='*70}")
    
    with open(database_path, 'r', encoding='utf-8') as f:
        database = json.load(f)
        
    judge = FinalJudge(
        threshold=Config.Refine.TARGET_ACCURACY,
        use_dino=True,
        use_clip=True,
        use_vqa=True
    )
    
    concept_dict = database.get("concept_dict", {})
    results = {}
    
    for concept_id, content in concept_dict.items():
        gen_image_path = os.path.join(output_dir, f"{concept_id}_generated.png")
        if not os.path.exists(gen_image_path):
            continue
            
        ref_image_path = content.get("image", content.get("selected_images", []))[0]
        fingerprints = content.get("info", {})
        
        try:
            judge_eval = judge.evaluate(
                generated_image=gen_image_path,
                reference_image=ref_image_path,
                fingerprints=fingerprints,
                prompt=fingerprints.get("sdxl_prompt", "") # O il nuovo flux_prompt se lo salviamo
            )
            results[concept_id] = judge_eval.to_dict()
            print(f"   {'✅' if judge_eval.passed else '❌'} {concept_id} | TIFA: {judge_eval.tifa_score:.1%} | DINO: {judge_eval.dino_i:.3f}")
        except Exception as e:
            print(f"   ⚠️ Errore su {concept_id}: {e}")
            
    judge.cleanup()
    
    # Salva risultati finali
    results_path = os.path.join(output_dir, "final_judge_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n📁 Risultati finali salvati in: {results_path}")

# =========================================================================
# MAIN ENTRY POINT
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R2P-GEN Modulare (FLUX Edition)")
    parser.add_argument("--database", type=str, required=True,
                       help="Path al database JSON")
    parser.add_argument("--output", type=str, default="output",
                       help="Directory di output")
    parser.add_argument("--stage", choices=["generate_only", "verify_base", "recovery", "final_judge", "full_auto"], 
                       required=True, help="Lo stadio della pipeline da eseguire")
    parser.add_argument("--num-shards", type=int, default=1, help="Numero totale di GPU/Worker")
    parser.add_argument("--shard-index", type=int, default=0, help="Indice di questo specifico Worker (0 a N-1)")
    
    args = parser.parse_args()
    
    if args.stage == "generate_only":
        stage_generate_only(args.database, args.output, args.num_shards, args.shard_index)
        
    elif args.stage == "verify_base":
        stage_verify_base(args.database, args.output)
        
    elif args.stage == "recovery":
        rejected_path = os.path.join(args.output, "rejected_concepts.json")
        stage_recovery(rejected_path, args.output)
        
    elif args.stage == "final_judge":
        stage_final_judge(args.database, args.output)
        
    elif args.stage == "full_auto":
        print("\n🚀 AVVIO PIPELINE FULL AUTO")
        stage_generate_only(args.database, args.output)
        rejected_path = stage_verify_base(args.database, args.output)
        stage_recovery(rejected_path, args.output)
        stage_final_judge(args.database, args.output)
        print("\n🏁 PIPELINE COMPLETATA")