import torch
import os
import re
import time
from PIL import Image
# Assicurati che questi import puntino ai tuoi file corretti
from pipeline2.r2p_tools import ClipScoreCalculator, ConfidenceCalculator, MiniCPMAdapter

# =============================================================================
# 1. MONKEY PATCH (FORZATA)
# =============================================================================
def _patch_minicpm_interface(reasoner):
    MiniCPMModelClass = reasoner.model_interface.__class__
    
    # 🚨 RIMOSSO IL CONTROLLO DI SICUREZZA PER FORZARE L'AGGIORNAMENTO 🚨
    # if getattr(MiniCPMModelClass, "_is_patched_for_v26", False):
    #     return

    print("🔧 Applying Monkey-Patch to MiniCPMModel.chat (Force Update)...")
    
    # Questa è la versione CORRETTA che accetta tokenizer e kwargs
    def patched_chat(self, msgs, tokenizer=None, **kwargs):
        # Se il tokenizer non è passato, usa quello dell'istanza
        use_tokenizer = tokenizer if tokenizer else self.tokenizer
        
        # Chiama il modello
        res = self.model.chat(
            msgs=msgs, 
            tokenizer=use_tokenizer,
            sampling=False, 
            max_new_tokens=10,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs 
        )
        
        if isinstance(res, tuple):
            if len(res) == 3: return res[2], res[0]
            elif len(res) == 2: return {}, res[0]
        
        if hasattr(res, 'scores'): return res, ""
        return {}, str(res)

    MiniCPMModelClass.chat = patched_chat
    MiniCPMModelClass._is_patched_for_v26 = True


# =============================================================================
# 2. INTERCEPTOR LOGIC
# =============================================================================
def chat_with_logits_interceptor(reasoner, msgs):
    model = reasoner.model_interface.model
    original_generate = model.generate
    captured_data = {"scores": None}

    def spy_generate(*args, **kwargs):
        kwargs['output_scores'] = True
        kwargs['return_dict_in_generate'] = True
        kwargs['max_new_tokens'] = 10 
        
        output = original_generate(*args, **kwargs)
        
        if hasattr(output, 'scores'):
            captured_data['scores'] = output.scores
        elif isinstance(output, dict) and 'scores' in output:
            captured_data['scores'] = output['scores']
        return output

    try:
        model.generate = spy_generate
        
        # Ora chiamiamo la patch che accetta ESPLICITAMENTE il tokenizer
        res = reasoner.model_interface.chat(
            msgs=msgs, 
            tokenizer=reasoner.model_interface.tokenizer, 
            sampling=False
        )
        
        response_text = ""
        if isinstance(res, tuple) and len(res) > 0:
            response_text = res[1] if len(res) >= 2 else str(res[0])
        elif isinstance(res, str):
            response_text = res
            
    except Exception as e:
        # Se fallisce ancora, stampiamo l'errore ma non crashiamo
        print(f"⚠️ Interceptor Error: {e}")
        response_text = "Error"
    finally:
        model.generate = original_generate

    class MockOutput: pass
    mock_out = MockOutput()
    mock_out.scores = captured_data['scores']
    
    return mock_out, response_text
    
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def call_model_with_scores(reasoner, image, prompt):
    """Helper che bypassa chat() e chiama generate() direttamente"""
    import torch
    
    # Prepara input (simplified per test rapido)
    full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = reasoner.model_interface.tokenizer(full_prompt, return_tensors="pt")
    inputs = {k: v.to(reasoner.model_interface.model.device) for k, v in inputs.items()}
    
    # Genera con scores
    with torch.no_grad():
        outputs = reasoner.model_interface.model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=reasoner.model_interface.tokenizer.eos_token_id
        )
    
    # Decodifica
    prompt_len = inputs['input_ids'].shape[1]
    generated_ids = outputs.sequences[0][prompt_len:]
    response_text = reasoner.model_interface.tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return outputs, response_text

def _extract_attributes_for_clip(fingerprints: dict) -> list:
    """Estrae gli attributi puliti mantenendo l'ordine di priorità."""
    attributes = []
    keys_to_check = ['distinct features', 'brand/text', 'pattern', 'material', 'color']
    seen = set()
    
    for key in keys_to_check:
        if key in fingerprints and fingerprints[key]:
            val = fingerprints[key]
            if "no visible" in val.lower() or "none" in val.lower(): continue
            
            chunks = re.split(r'[;.]|\n', val)
            for chunk in chunks:
                clean_chunk = chunk.strip()
                if len(clean_chunk) > 4 and "image" not in clean_chunk.lower():
                    if clean_chunk not in seen:
                        attributes.append(clean_chunk)
                        seen.add(clean_chunk)
    
    if not attributes and 'attributes' in fingerprints:
         raw_attrs = []
         if isinstance(fingerprints['attributes'], list):
             raw_attrs = [str(v) for v in fingerprints['attributes']]
         elif isinstance(fingerprints['attributes'], dict):
             raw_attrs = [str(v) for v in fingerprints['attributes'].values()]
         
         for attr in raw_attrs:
             if attr not in seen:
                 attributes.append(attr)
                 seen.add(attr)

    return attributes 
    
# =============================================================================
# MAIN VERIFICATION LOGIC
# =============================================================================

import torch
import os
import re
import numpy as np
from PIL import Image
from pipeline2.r2p_tools import ClipScoreCalculator, ConfidenceCalculator, MiniCPMAdapter

# =============================================================================
# MONKEY PATCHING (Invariato)
# =============================================================================
def _patch_minicpm_interface(reasoner):
    MiniCPMModelClass = reasoner.model_interface.__class__
    if getattr(MiniCPMModelClass, "_is_patched_for_v26", False):
        return
    print("🔧 Applying Monkey-Patch to MiniCPMModel.chat for v2.6 compatibility...")
    def patched_chat(self, msgs):
        res = self.model.chat(msgs=msgs, tokenizer=self.tokenizer)
        if isinstance(res, tuple):
            if len(res) == 3:
                answer, history, extra = res
                return extra, answer
            elif len(res) == 2:
                answer, history = res
                return {}, answer
        return {}, res
    MiniCPMModelClass.chat = patched_chat
    MiniCPMModelClass._is_patched_for_v26 = True

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def _extract_attributes_for_clip(fingerprints: dict) -> list:
    """Estrae TUTTI gli attributi puliti."""
    attributes = []
    keys_to_check = ['distinct features', 'brand/text', 'pattern', 'material', 'color']
    seen = set()
    
    for key in keys_to_check:
        if key in fingerprints and fingerprints[key]:
            val = fingerprints[key]
            if "no visible" in val.lower() or "none" in val.lower(): continue
            chunks = re.split(r'[;.]|\n', val)
            for chunk in chunks:
                clean_chunk = chunk.strip()
                if len(clean_chunk) > 4 and "image" not in clean_chunk.lower():
                    if clean_chunk not in seen:
                        attributes.append(clean_chunk)
                        seen.add(clean_chunk)
    
    # Fallback
    if not attributes and 'attributes' in fingerprints:
         raw_attrs = []
         if isinstance(fingerprints['attributes'], list):
             raw_attrs = [str(v) for v in fingerprints['attributes']]
         elif isinstance(fingerprints['attributes'], dict):
             raw_attrs = [str(v) for v in fingerprints['attributes'].values()]
         for attr in raw_attrs:
             if attr not in seen:
                 attributes.append(attr)
                 seen.add(attr)

    return attributes 

# =============================================================================
# MAIN LOGIC v3: DISAGREEMENT-BASED PIPELINE
# =============================================================================

def verify_generation_r2p(
    reasoner, 
    clip_calculator: ClipScoreCalculator, 
    gen_image_path: str, 
    ref_image_path: str, 
    fingerprints: dict,
    # SOGLIE DI SICUREZZA
    vlm_high_confidence: float = 0.85,  # Sopra questo -> probabile pass
    vlm_low_confidence: float = 0.40,   # Sotto questo -> probabile fail
    clip_hard_floor: float = 0.15,
    max_drop_threshold: float = -0.03,
):
    # 0. SETUP
    _patch_minicpm_interface(reasoner)
    vlm_history = []
    clip_details = {'gen': {}, 'ref': {}}
    
    # Init Conf Calculator
    conf_calc = ConfidenceCalculator(reasoner.model_interface.tokenizer)
    
    print(f"\n--- Starting R2P Verification v3 for {os.path.basename(gen_image_path)} ---")
    
    try:
        gen_image = Image.open(gen_image_path).convert("RGB")
        ref_image = Image.open(ref_image_path).convert("RGB")
    except Exception as e:
        return {"is_verified": False, "reason": f"Image error: {e}", "vlm_history": [], "method": "Error"}

    # Prep Attributes (TUTTI, non solo i primi 4)
    attributes_list = _extract_attributes_for_clip(fingerprints)
    
    # =========================================================================
    # PHASE 1: VLM SINGLE CHECK (Full Attribute Sweep with Confidence)
    # =========================================================================
    print(f"[Phase 1] VLM Sweep on {len(attributes_list)} attributes...")

    vlm_scores = []

    for attr in attributes_list:
        # Prompt che forza una risposta binaria misurabile
        prompt = (
            f"Look at the image. Is the feature '{attr}' clearly visible and correct?\n"
            f"Ignore minor style issues, focus on the presence of the feature.\n"
            f"Answer strictly Yes or No."
        )
        
        # 1. Chiamata al modello (Usa la chat wrappata che gestisce l'immagine correttamente)
        msgs = [{"role": "user", "content": [gen_image, prompt]}]
        outputs, response_text = chat_with_logits_interceptor(reasoner, msgs)
                
        
       # Controllo di sicurezza prima di stampare
        if outputs.scores is not None:
            print(f"   Scores captured: {len(outputs.scores)} tokens")
        else:
            print("   ⚠️ WARNING: No scores captured by interceptor.")

        # ⭐ DEBUG: Aggiungi questo blocco
        print(f"\n🔍 DEBUG OUTPUTS:")
        print(f"   Type: {type(outputs)}")
        print(f"   Is dict: {isinstance(outputs, dict)}")
        if isinstance(outputs, dict):
            print(f"   Keys: {list(outputs.keys())}")
        else:
            print(f"   Has attr 'scores': {hasattr(outputs, 'scores')}")
            print(f"   Has attr 'logits': {hasattr(outputs, 'logits')}")
            print(f"   Dir: {[x for x in dir(outputs) if not x.startswith('_')][:10]}")

        # Se esiste scores/logits, mostra la struttura
        if isinstance(outputs, dict) and 'scores' in outputs:
            print(f"   Scores type: {type(outputs['scores'])}")
            print(f"   Scores len: {len(outputs['scores']) if outputs['scores'] else 0}")
        elif hasattr(outputs, 'scores'):
            print(f"   Scores type: {type(outputs.scores)}")
            print(f"   Scores len: {len(outputs.scores) if outputs.scores else 0}")
        print(f"   Response text: '{response_text[:50]}'...\n")

        # 2. ⭐ NUOVO: Calcolo Confidence usando il metodo corretto
        result = conf_calc.calculate_binary_confidence(outputs, response_text)
        
        # Usa yes_confidence come score (più alto = più sicuro che la feature sia presente)
        score = result['yes_confidence']
        
        vlm_scores.append(score)
        
        vlm_history.append({
            "phase": "single_check",
            "attribute": attr,
            "prompt": prompt,
            "response": response_text,
            "score": score,
            "method": result['method'],  # Per debugging
            "yes_conf": result['yes_confidence'],
            "no_conf": result['no_confidence']
        })
        
        # Emoji basato su confidence reale
        emoji = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴"
        print(f"   - {attr[:25]:<25} | Y:{result['yes_confidence']:.2f} N:{result['no_confidence']:.2f} {emoji} [{result['method']}]")

    # CALCOLO MEDIA VLM (Mancante nel tuo codice originale)
    if vlm_scores:
        vlm_avg_score = sum(vlm_scores) / len(vlm_scores)
    else:
        vlm_avg_score = 0.0
    
    print(f"📊 VLM Average Score: {vlm_avg_score:.4f}")

    # =========================================================================
    # PHASE 2: CLIP CHECK
    # =========================================================================
    gen_clip_score, gen_breakdown = clip_calculator.compute_attribute_score(gen_image, attributes_list)
    _, ref_breakdown = clip_calculator.compute_attribute_score(ref_image, attributes_list)
    clip_details = {'gen': gen_breakdown, 'ref': ref_breakdown}
    
    # Logica Drop
    clip_pass = True
    clip_issues = []
    
    # Check 1: Score assoluto troppo basso
    if gen_clip_score < clip_hard_floor:
        clip_pass = False
        clip_issues.append(f"Low Avg Score ({gen_clip_score:.2f})")

    # Check 2: Significant Drops
    for attr, ref_val in ref_breakdown.items():
        if ref_val > 0.18: # Solo se rilevante nell'originale
            delta = gen_breakdown.get(attr, 0) - ref_val
            if delta < max_drop_threshold:
                clip_pass = False
                clip_issues.append(f"Drop in '{attr}' ({delta:.3f})")

    print(f"[Phase 2] CLIP Check: {'PASS' if clip_pass else 'FAIL'} (Issues: {clip_issues})")

    # =========================================================================
    # PHASE 3: DECISION GATE (The "Early Exit" Logic)
    # =========================================================================
    
    # 1. STRONG AGREEMENT -> PASS
    if vlm_avg_score >= vlm_high_confidence and clip_pass:
        print("[Gate] ✅ AUTO-PASS: Strong Agreement (VLM High + CLIP Pass)")
        return {
            "is_verified": True,
            "score": vlm_avg_score,
            "method": "AutoPass_Agreement",
            "reason": "VLM and CLIP strongly agree on quality.",
            "vlm_history": vlm_history,
            "clip_details": clip_details
        }

    # 2. STRONG AGREEMENT -> FAIL (Refine)
    if vlm_avg_score <= vlm_low_confidence and not clip_pass:
        print("[Gate] ❌ AUTO-FAIL: Strong Agreement (VLM Low + CLIP Fail)")
        return {
            "is_verified": False,
            "score": vlm_avg_score,
            "method": "AutoFail_Agreement",
            "reason": f"Both models rejected. CLIP issues: {clip_issues}",
            "vlm_history": vlm_history,
            "clip_details": clip_details
        }
        
    # 3. DISAGREEMENT / UNCERTAINTY -> TRIGGER PAIRWISE
    reason_for_trigger = []
    if not clip_pass: reason_for_trigger.append("CLIP detected drops")
    if vlm_avg_score < vlm_high_confidence: reason_for_trigger.append(f"VLM uncertain ({vlm_avg_score:.2f})")
    
    print(f"[Gate] ⚠️ DISAGREEMENT DETECTED: {', '.join(reason_for_trigger)}")
    print("       >>> TRIGGERING PHASE 4 (Pairwise Analysis) on ALL attributes.")

    # =========================================================================
    # PHASE 4: PAIRWISE REASONING (Only if needed)
    # =========================================================================
    
    adapter = MiniCPMAdapter(reasoner.model_interface.tokenizer)
    pairwise_scores = []
    
    # Iteriamo su TUTTI gli attributi perché siamo in fase di arbitraggio
    for attr in attributes_list:
        prompt_text = (
            f"Compare the specific feature '{attr}' in Image 1 and Image 2.\n"
            f"Is this feature present and visually consistent in both images?\n"
            f"Answer Yes or No."
        )
        
        msgs = adapter.format_image2image_plus_text_comparison_msgs(gen_image, ref_image, prompt_text)
        outputs, response_text = chat_with_logits_interceptor(reasoner, msgs)        
        # ⭐ NUOVO: Usa il metodo corretto
        result = conf_calc.calculate_binary_confidence(outputs, response_text)
        p_yes = result['yes_confidence']
        
        pairwise_scores.append(p_yes)
        
        vlm_history.append({
            "phase": "pairwise",
            "attribute": attr,
            "score": p_yes,
            "method": result['method'],
            "yes_conf": result['yes_confidence'],
            "no_conf": result['no_confidence']
        })
        
        emoji = "✅" if p_yes > 0.6 else "⚠️"
        print(f"   - Pairwise '{attr[:20]}': Y:{result['yes_confidence']:.2f} {emoji} [{result['method']}]")

    # Calcolo Score Finale Pairwise
    final_pairwise_score = np.mean(pairwise_scores) if pairwise_scores else 0.0
    
    # Verdetto Finale basato SOLO su Pairwise (l'arbitro finale)
    PAIRWISE_THRESHOLD = 0.60 # Soglia media per l'accettazione finale
    
    is_verified = final_pairwise_score >= PAIRWISE_THRESHOLD
    
    status = "Pass" if is_verified else "Fail"
    print(f"[Phase 4] Final Pairwise Score: {final_pairwise_score:.4f} -> {status}")

    return {
        "is_verified": is_verified,
        "score": final_pairwise_score,
        "method": "Arbitration_Pairwise",
        "reason": f"Disagreement resolved by Pairwise. Score: {final_pairwise_score:.2f}",
        "vlm_history": vlm_history,
        "clip_details": clip_details
    }