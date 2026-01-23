import torch
import os
import re
import time
import numpy as np  # <--- FIX: Aggiunto import mancante
from PIL import Image
from pipeline2.r2p_tools import ClipScoreCalculator, ConfidenceCalculator, MiniCPMAdapter

# =============================================================================
# SEZIONE 1: MONKEY PATCH (ROBUSTA & SMART)
# =============================================================================
def _patch_minicpm_interface(reasoner):
    """
    Modifica la funzione chat del modello per accettare kwargs extra
    senza causare conflitti o errori di argomenti duplicati.
    """
    MiniCPMModelClass = reasoner.model_interface.__class__
    
    # Rimuovi il commento qui sotto se vuoi evitare di ri-applicare la patch ogni volta
    # if getattr(MiniCPMModelClass, "_is_patched_for_v26", False):
    #     return

    print("🔧 Applying Monkey-Patch to MiniCPMModel.chat (Smart Merge Version)...")
    
    def patched_chat(self, msgs, tokenizer=None, **kwargs):
        # 1. Gestione Tokenizer: usa quello passato o quello dell'istanza
        use_tokenizer = tokenizer if tokenizer else self.tokenizer
        
        # 2. Definisci i parametri di default che servono per i logits
        # Questi sono i parametri BASE che vogliamo sempre.
        params = {
            "sampling": False,
            "max_new_tokens": 10,
            "return_dict_in_generate": True,
            "output_scores": True
        }
        
        # 3. Unisci con kwargs: se l'interceptor passa parametri (es. 'sampling'),
        # questi sovrascriveranno i default senza creare duplicati.
        params.update(kwargs)
        
        # 4. Chiama il modello passando il dizionario unificato
        # Questo evita l'errore "multiple values for keyword argument"
        res = self.model.chat(
            msgs=msgs, 
            tokenizer=use_tokenizer,
            **params 
        )
        
        # 5. Gestione Output (tuple vs object vs string)
        if isinstance(res, tuple):
            if len(res) == 3: return res[2], res[0]
            elif len(res) == 2: return {}, res[0]
        
        if hasattr(res, 'scores'): return res, ""
        return {}, str(res)

    MiniCPMModelClass.chat = patched_chat
    MiniCPMModelClass._is_patched_for_v26 = True


# =============================================================================
# SEZIONE 2: INTERCEPTOR LOGIC (GENERATION CONFIG AWARE)
# =============================================================================
def chat_with_logits_interceptor(reasoner, msgs):
    """
    Intercetta la chiamata a generate() per rubare i logits (scores).
    FIX CRITICO: Modifica anche generation_config se presente, 
    altrimenti i kwargs vengono ignorati dal modello.
    """
    model = reasoner.model_interface.model
    original_generate = model.generate
    captured_data = {"scores": None}

    def spy_generate(*args, **kwargs):
        # 1. FORZA I KWARGS BASE
        kwargs['output_scores'] = True
        kwargs['return_dict_in_generate'] = True
        kwargs['max_new_tokens'] = 10
        
        # 2. FIX CRITICO: HACK DI GENERATION_CONFIG
        # MiniCPM crea un oggetto GenerationConfig che ha la priorità sui kwargs.
        # Dobbiamo intercettarlo e modificarlo brutalmente.
        
        # Cerca nei kwargs
        if 'generation_config' in kwargs:
            gen_config = kwargs['generation_config']
            gen_config.output_scores = True
            gen_config.return_dict_in_generate = True
            # Assicuriamoci che non stia facendo sampling se non vogliamo
            if 'sampling' in kwargs and not kwargs['sampling']:
                 gen_config.do_sample = False
        
        # Cerca negli args (di solito è il 3° o 4° argomento posizionale, ma cerchiamo per tipo)
        from transformers import GenerationConfig
        for arg in args:
            if isinstance(arg, GenerationConfig):
                arg.output_scores = True
                arg.return_dict_in_generate = True

        # 3. Chiamata reale al modello
        output = original_generate(*args, **kwargs)
        
        # 4. Cattura dei dati
        # Se output è un oggetto ModelOutput (comportamento corretto con return_dict=True)
        if hasattr(output, 'scores'):
            captured_data['scores'] = output.scores
        elif isinstance(output, dict) and 'scores' in output:
            captured_data['scores'] = output['scores']
        # Fallback: Se per qualche motivo ritorna ancora solo tensori (tuple), non possiamo farci nulla
        
        return output

    try:
        # Sostituiamo generate
        model.generate = spy_generate
        
        # Chiamiamo la chat patchata
        # Passiamo sampling=False esplicitamente per aiutare la config
        res = reasoner.model_interface.chat(
            msgs=msgs, 
            tokenizer=reasoner.model_interface.tokenizer, 
            sampling=False 
        )
        
        # Estrazione testo risposta
        response_text = ""
        if isinstance(res, tuple) and len(res) > 0:
            response_text = res[1] if len(res) >= 2 else str(res[0])
        elif isinstance(res, str):
            response_text = res
            
    except Exception as e:
        print(f"⚠️ Interceptor Error: {e}")
        response_text = "Error"
    finally:
        # Ripristino
        model.generate = original_generate

    # Costruiamo l'output per il calcolatore
    class MockOutput: pass
    mock_out = MockOutput()
    mock_out.scores = captured_data['scores']
    
    return mock_out, response_text
    
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
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

    # Prep Attributes (TUTTI)
    attributes_list = _extract_attributes_for_clip(fingerprints)
    
    # =========================================================================
    # PHASE 1: VLM SINGLE CHECK (Full Attribute Sweep with Confidence)
    # =========================================================================
    print(f"[Phase 1] VLM Sweep on {len(attributes_list)} attributes...")

    vlm_scores = []

    for attr in attributes_list:
        prompt = (
            f"Look at the image. Is the feature '{attr}' clearly visible and correct?\n"
            f"Ignore minor style issues, focus on the presence of the feature.\n"
            f"Answer strictly Yes or No."
        )
        
        # 1. Chiamata al modello tramite Interceptor
        msgs = [{"role": "user", "content": [gen_image, prompt]}]
        outputs, response_text = chat_with_logits_interceptor(reasoner, msgs)
        
        # Debugging Output (Solo se servono controlli approfonditi)
        # if outputs.scores is None:
        #    print("   ⚠️ WARNING: No scores captured. Check interceptor logic.")

        # 2. Calcolo Confidence
        result = conf_calc.calculate_binary_confidence(outputs, response_text)
        
        score = result['yes_confidence']
        vlm_scores.append(score)
        
        vlm_history.append({
            "phase": "single_check",
            "attribute": attr,
            "prompt": prompt,
            "response": response_text,
            "score": score,
            "method": result['method'],
            "yes_conf": result['yes_confidence'],
            "no_conf": result['no_confidence']
        })
        
        emoji = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴"
        print(f"   - {attr[:25]:<25} | Y:{result['yes_confidence']:.2f} N:{result['no_confidence']:.2f} {emoji} [{result['method']}]")

    # CALCOLO MEDIA VLM
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
    
    clip_pass = True
    clip_issues = []
    
    if gen_clip_score < clip_hard_floor:
        clip_pass = False
        clip_issues.append(f"Low Avg Score ({gen_clip_score:.2f})")

    for attr, ref_val in ref_breakdown.items():
        if ref_val > 0.18: 
            delta = gen_breakdown.get(attr, 0) - ref_val
            if delta < max_drop_threshold:
                clip_pass = False
                clip_issues.append(f"Drop in '{attr}' ({delta:.3f})")

    print(f"[Phase 2] CLIP Check: {'PASS' if clip_pass else 'FAIL'} (Issues: {clip_issues})")

    # =========================================================================
    # PHASE 3: DECISION GATE
    # =========================================================================
    
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
        
    reason_for_trigger = []
    if not clip_pass: reason_for_trigger.append("CLIP detected drops")
    if vlm_avg_score < vlm_high_confidence: reason_for_trigger.append(f"VLM uncertain ({vlm_avg_score:.2f})")
    
    print(f"[Gate] ⚠️ DISAGREEMENT DETECTED: {', '.join(reason_for_trigger)}")
    print("       >>> TRIGGERING PHASE 4 (Pairwise Analysis) on ALL attributes.")

    # =========================================================================
    # PHASE 4: PAIRWISE REASONING
    # =========================================================================
    
    adapter = MiniCPMAdapter(reasoner.model_interface.tokenizer)
    pairwise_scores = []
    
    for attr in attributes_list:
        prompt_text = (
            f"Compare the specific feature '{attr}' in Image 1 and Image 2.\n"
            f"Is this feature present and visually consistent in both images?\n"
            f"Answer Yes or No."
        )
        
        msgs = adapter.format_image2image_plus_text_comparison_msgs(gen_image, ref_image, prompt_text)
        outputs, response_text = chat_with_logits_interceptor(reasoner, msgs)        
        
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

    # FIX CRASH: Usa np importato correttamente
    final_pairwise_score = np.mean(pairwise_scores) if pairwise_scores else 0.0
    
    PAIRWISE_THRESHOLD = 0.60 
    
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