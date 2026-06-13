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
import openai
from pipeline.recovery import vlm_rewrite_prompt, generate_recovery_http
from pipeline.prompts.flux_prompts import build_flux_prompt
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

def stage_recovery(database_path: str, rejected_path: str, output_dir: str):
    """Fase 4: Recovery tramite API (Il Closed-Loop Ospedaliero)"""
    print(f"\n{'='*70}\n🚑 STAGE: RECOVERY API\n{'='*70}")
    if not os.path.exists(rejected_path):
        print(f"   ✅ Nessun file rejected trovato. Niente da recuperare!")
        return
        
    with open(rejected_path, 'r', encoding='utf-8') as f:
        rejected_dict = json.load(f)
        
    if not rejected_dict:
        print("   ✅ Lista rejected vuota. Niente da recuperare!")
        return
        
    # Carichiamo il database originale per avere le immagini sorgenti e le info
    with open(database_path, 'r', encoding='utf-8') as f:
        database = json.load(f)
    concept_dict = database.get("concept_dict", {})

    print(f"   Trovati {len(rejected_dict)} concetti da curare.")
    
    # Setup dei Client API (Devono essere attivi sui rispettivi server!)
    # NOTA: Regola questi parametri o mettili nel tuo config.py
    VLM_URL = "http://localhost:8000/v1" # Es. server vLLM per QwenVL3
    VLM_MODEL_NAME = "Qwen/Qwen-VL-Chat" # Sostituisci con il tuo nome modello esatto
    FLUX_URL = "http://localhost:8766"   # Server FastAPI FLUX
    MAX_ATTEMPTS = 4
    
    try:
        vlm_client = openai.OpenAI(api_key="EMPTY", base_url=VLM_URL)
    except Exception as e:
        print(f"   ❌ Errore inizializzazione client VLM: {e}")
        return

    # Inizializziamo i modelli di verifica per il "Controllo Medico" post-cura
    # NOTA: Se usi Qwen anche per la verifica base, qui dovrai instanziare quello!
    # Per coerenza con il resto, supponiamo tu abbia ancora il valutatore pronto.
    print("   Caricamento modelli di verifica per il loop di recovery...")
    reasoner = MiniCPMReasoning(model_path=Config.Models.VLM_MODEL, device="cuda", attn_implementation="sdpa", seed=Config.Generate.SEED)
    clip_calculator = ClipScoreCalculator(device="cuda")
    
    recovered_count = 0
    graveyard_count = 0

    for concept_id, fail_data in rejected_dict.items():
        print(f"\n   🚑 Paziente: {concept_id} | Problema: {fail_data.get('missing_details', ['Unknown'])}")
        
        content = concept_dict.get(concept_id, {})
        source_image_path = content.get("image", content.get("selected_images", []))[0]
        fingerprints = content.get("info", {})
        
        # Recuperiamo il prompt base
        target_context = Config.get_background_template()
        current_prompt = build_flux_prompt(fingerprints, target_context)
        missing_details = fail_data.get("missing_details", [])
        
        gen_image_path = os.path.join(output_dir, f"{concept_id}_generated.png")
        is_fixed = False

        for attempt in range(1, MAX_ATTEMPTS + 1):
            print(f"      🔄 Tentativo {attempt}/{MAX_ATTEMPTS}...")
            
            # 1. Prescrizione (Rewrite Prompt)
            new_prompt = vlm_rewrite_prompt(
                vlm_client=vlm_client, 
                model_name=VLM_MODEL_NAME, 
                original_prompt=current_prompt, 
                missing_details=missing_details, 
                attempt=attempt
            )
            print(f"         📝 Nuovo prompt: {new_prompt[:80]}...")
            
            # 2. Terapia (Rigenerazione con FLUX API)
            new_seed = Config.Generate.SEED + attempt # Variazione vitale del seed!
            success = generate_recovery_http(
                flux_url=FLUX_URL,
                source_image_path=source_image_path,
                prompt=new_prompt,
                seed=new_seed,
                output_path=gen_image_path # Sovrascrive la vecchia immagine
            )
            
            if not success:
                print("         ⚠️ Rigenerazione fallita a livello API, passo al prossimo tentativo.")
                continue
                
            # 3. Visita di Controllo (Verifica)
            verification = verify_generation_r2p(
                reasoner=reasoner,
                clip_calculator=clip_calculator,
                gen_image_path=gen_image_path,
                ref_image_path=source_image_path,
                fingerprints=fingerprints
            )
            
            if verification["is_verified"]:
                print(f"         ✅ GUARITO! L'immagine ha superato i controlli al tentativo {attempt}.")
                is_fixed = True
                recovered_count += 1
                break
            else:
                # Se fallisce di nuovo, aggiorniamo i dettagli mancanti per il prossimo loop
                missing_details = verification.get('failed_attributes', missing_details)
                current_prompt = new_prompt # Il VLM modificherà l'ultimo prompt generato
                print(f"         ❌ Visita fallita. Permangono problemi: {missing_details}")
                
        if not is_fixed:
            print(f"      💀 UNRECOVERABLE: {concept_id} trasferito nel Graveyard.")
            graveyard_count += 1
            
    print(f"\n📊 Summary Recovery: {recovered_count} Salvati | {graveyard_count} Graveyard")
    
    # Pulizia
    del reasoner
    del clip_calculator
    cleanup_gpu()

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