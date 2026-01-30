import torch
import os
import re
import numpy as np
from PIL import Image
from pipeline2.r2p_tools import ClipScoreCalculator, LLMConfidenceExtractor, MiniCPMAdapter

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
    # CONFIDENCE THRESHOLDS
    vlm_high_confidence: float = 0.85,
    vlm_low_confidence: float = 0.40,
    clip_hard_floor: float = 0.15,
    max_drop_threshold: float = -0.03,
):
    """
    R2P Verification Pipeline V4.
    
    Uses LLMConfidenceExtractor for reliable logit-based confidence.
    
    Pipeline:
        Phase 1: VLM Single Check (attribute sweep on generated image)
        Phase 2: CLIP Check (cross-modal verification)
        Phase 3: Decision Gate (auto-pass/fail or trigger pairwise)
        Phase 4: Pairwise Reasoning (if needed)
    """
    
    vlm_history = []
    clip_details = {'gen': {}, 'ref': {}}
    
    # Initialize the LLM-level confidence extractor (THE FIX!)
    conf_extractor = LLMConfidenceExtractor(reasoner.model_interface)
    
    print(f"\n--- Starting R2P Verification V4 for {os.path.basename(gen_image_path)} ---")
    
    # Load images
    try:
        gen_image = Image.open(gen_image_path).convert("RGB")
        ref_image = Image.open(ref_image_path).convert("RGB")
    except Exception as e:
        return {"is_verified": False, "reason": f"Image error: {e}", "vlm_history": [], "method": "Error"}

    # Extract attributes
    attributes_list = _extract_attributes_for_clip(fingerprints)
    
    if not attributes_list:
        print("⚠️ No attributes found in fingerprints!")
        return {
            "is_verified": False,
            "reason": "No attributes to verify",
            "vlm_history": [],
            "method": "Error_NoAttributes"
        }
    
    # =========================================================================
    # PHASE 1: VLM SINGLE CHECK (Attribute Sweep with Real Logits)
    # =========================================================================
    print(f"[Phase 1] VLM Sweep on {len(attributes_list)} attributes...")

    vlm_scores = []

    for attr in attributes_list:
        prompt = (
            f"Look at the image. Is the feature '{attr}' clearly visible and correct?\n"
            f"Ignore minor style issues, focus on the presence of the feature.\n"
            f"Answer strictly Yes or No."
        )
        
        # Query with confidence extraction (using LLM-level interception)
        confidence, response_text = conf_extractor.query_with_confidence(
            image=gen_image,
            prompt=prompt,
            max_new_tokens=5
        )
        
        score = confidence['yes_confidence']
        vlm_scores.append(score)
        
        vlm_history.append({
            "phase": "single_check",
            "attribute": attr,
            "prompt": prompt,
            "response": response_text,
            "score": score,
            "method": confidence.get('method', 'unknown'),
            "yes_conf": confidence['yes_confidence'],
            "no_conf": confidence['no_confidence']
        })
        
        # Status emoji based on confidence method
        if confidence.get('method') == 'logits':
            emoji = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴"
        else:
            emoji = "⚠️"  # Text fallback indicator
            
        print(f"   - {attr[:25]:<25} | Y:{confidence['yes_confidence']:.2f} N:{confidence['no_confidence']:.2f} {emoji} [{confidence.get('method', '?')}]")

    # Calculate VLM average
    vlm_avg_score = sum(vlm_scores) / len(vlm_scores) if vlm_scores else 0.0
    
    # Log method distribution
    logit_count = sum(1 for h in vlm_history if h.get('method') == 'logits')
    print(f"📊 VLM Average Score: {vlm_avg_score:.4f} (Logit method: {logit_count}/{len(vlm_history)})")

    # =========================================================================
    # PHASE 2: CLIP CHECK
    # =========================================================================
    print(f"[Phase 2] CLIP Verification...")
    
    gen_clip_score, gen_breakdown = clip_calculator.compute_attribute_score(gen_image, attributes_list)
    _, ref_breakdown = clip_calculator.compute_attribute_score(ref_image, attributes_list)
    clip_details = {'gen': gen_breakdown, 'ref': ref_breakdown}
    
    clip_pass = True
    clip_issues = []
    
    if gen_clip_score < clip_hard_floor:
        clip_pass = False
        clip_issues.append(f"Low Avg Score ({gen_clip_score:.2f})")

    for attr, ref_val in ref_breakdown.items():
        if ref_val > 0.18:  # Only check significant attributes
            delta = gen_breakdown.get(attr, 0) - ref_val
            if delta < max_drop_threshold:
                clip_pass = False
                clip_issues.append(f"Drop in '{attr[:20]}' ({delta:.3f})")

    print(f"   CLIP Score: {gen_clip_score:.4f} | {'PASS' if clip_pass else 'FAIL'}")
    if clip_issues:
        print(f"   Issues: {clip_issues[:3]}...")  # Show first 3

    # =========================================================================
    # PHASE 3: DECISION GATE
    # =========================================================================
    print(f"[Phase 3] Decision Gate...")
    
    # Case 1: Strong agreement - AUTO-PASS
    if vlm_avg_score >= vlm_high_confidence and clip_pass:
        print("   ✅ AUTO-PASS: Strong Agreement (VLM High + CLIP Pass)")
        return {
            "is_verified": True,
            "score": vlm_avg_score,
            "method": "AutoPass_Agreement",
            "reason": "VLM and CLIP strongly agree on quality.",
            "vlm_history": vlm_history,
            "clip_details": clip_details
        }

    # Case 2: Strong agreement - AUTO-FAIL
    if vlm_avg_score <= vlm_low_confidence and not clip_pass:
        print("   ❌ AUTO-FAIL: Strong Agreement (VLM Low + CLIP Fail)")
        return {
            "is_verified": False,
            "score": vlm_avg_score,
            "method": "AutoFail_Agreement",
            "reason": f"Both models rejected. CLIP issues: {clip_issues}",
            "vlm_history": vlm_history,
            "clip_details": clip_details
        }
    
    # Case 3: Disagreement - Trigger Pairwise
    reason_for_trigger = []
    if not clip_pass: 
        reason_for_trigger.append("CLIP detected drops")
    if vlm_avg_score < vlm_high_confidence: 
        reason_for_trigger.append(f"VLM uncertain ({vlm_avg_score:.2f})")
    
    print(f"   ⚠️ DISAGREEMENT: {', '.join(reason_for_trigger)}")
    print("   >>> TRIGGERING PHASE 4 (Pairwise Analysis)")

    # =========================================================================
    # PHASE 4: PAIRWISE REASONING
    # =========================================================================
    print(f"[Phase 4] Pairwise Comparison on {len(attributes_list)} attributes...")
    
    pairwise_scores = []
    
    for attr in attributes_list:
        prompt_text = (
            f"Compare the specific feature '{attr}' in Image 1 and Image 2.\n"
            f"Is this feature present and visually consistent in both images?\n"
            f"Answer Yes or No."
        )
        
        # Pairwise query with both images
        confidence, response_text = conf_extractor.query_with_confidence(
            image=gen_image,
            prompt=prompt_text,
            image2=ref_image,
            max_new_tokens=5
        )
        
        p_yes = confidence['yes_confidence']
        pairwise_scores.append(p_yes)
        
        vlm_history.append({
            "phase": "pairwise",
            "attribute": attr,
            "score": p_yes,
            "method": confidence.get('method', 'unknown'),
            "yes_conf": confidence['yes_confidence'],
            "no_conf": confidence['no_confidence']
        })
        
        emoji = "✅" if p_yes > 0.6 else "⚠️" if p_yes > 0.4 else "❌"
        print(f"   - {attr[:20]:<20} | Y:{confidence['yes_confidence']:.2f} {emoji} [{confidence.get('method', '?')}]")

    # Final decision
    final_pairwise_score = np.mean(pairwise_scores) if pairwise_scores else 0.0
    
    PAIRWISE_THRESHOLD = 0.60
    is_verified = final_pairwise_score >= PAIRWISE_THRESHOLD
    
    status = "✅ PASS" if is_verified else "❌ FAIL"
    print(f"   Final Pairwise Score: {final_pairwise_score:.4f} -> {status}")

    return {
        "is_verified": is_verified,
        "score": final_pairwise_score,
        "method": "Arbitration_Pairwise",
        "reason": f"Disagreement resolved by Pairwise. Score: {final_pairwise_score:.2f}",
        "vlm_history": vlm_history,
        "clip_details": clip_details
    }