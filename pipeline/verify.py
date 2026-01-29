# verify.py
import sys
import os
import torch
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'r2p_core'))

from models.mini_cpm_reasoning import MiniCPMReasoning
from config import Config
from pipeline.utils2 import cleanup_gpu

def verify_generation_detailed(generated_image_path, target_fingerprints, model=None):
    """
    Verifica GRANULARE: controlla ogni attributo separatamente
    
    Returns:
        dict: {
            "missing": [(attr_key, attr_value), ...],
            "present": [attr_key, ...],
            "accuracy": float (0.0-1.0),
            "details": {attr_key: response_text}
        }
    """
    print(f"\n⚖️ [VERIFY] Verifica Granulare Attributi")
    
    # Carica modello se necessario
    if model is None:
        print("   -> Caricamento VLM Judge...")
        model = MiniCPMReasoning(
            model_path=Config.VLM_MODEL,
            device="cuda",
            torch_dtype=torch.float16 if Config.USE_FP16 else torch.float32,
            attn_implementation="sdpa",
            seed=Config.SEED
        )
    
    gen_img = Image.open(generated_image_path).convert('RGB')
    
    # Ridimensiona se necessario
    if max(gen_img.size) > Config.MAX_IMAGE_DIM:
        ratio = Config.MAX_IMAGE_DIM / max(gen_img.size)
        new_size = tuple(int(dim * ratio) for dim in gen_img.size)
        gen_img = gen_img.resize(new_size, Image.Resampling.LANCZOS)
    
    missing_attributes = []
    present_attributes = []
    details = {}
    
    # Filtra attributi da verificare
    attrs_to_check = {k: v for k, v in target_fingerprints.items() 
                      if k != "description" and v}
    
    if not attrs_to_check:
        print("   ⚠️ Nessun attributo strutturato da verificare")
        # Fallback su verifica descrizione generale
        return verify_description_fallback(gen_img, target_fingerprints, model)
    
    print(f"   -> Verifica di {len(attrs_to_check)} attributi...")
    
    # Verifica ogni attributo
    for attr_key, attr_value in attrs_to_check.items():
        print(f"   🔍 Checking '{attr_key}': {attr_value[:50]}...")
        
        # Prompt specifico per attributo
        msgs = [{
            "role": "user",
            "content": f"Look at this image. Does it clearly show: {attr_key} = '{attr_value}'? Answer ONLY 'Yes' or 'No' first, then explain briefly in one sentence."
        }]
        
        try:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    result = model.model_interface.model.chat(
                        image=gen_img,
                        msgs=msgs,
                        tokenizer=model.model_interface.tokenizer
                    )
                    
                    response = result[-1] if isinstance(result, tuple) else str(result)
                    details[attr_key] = response
            
            # Parse risposta
            response_lower = response.lower()
            is_present = any(word in response_lower[:50] for word in ["yes", "correct", "present", "shows"])
            
            if is_present:
                present_attributes.append(attr_key)
                print(f"      ✅ PRESENTE")
            else:
                missing_attributes.append((attr_key, attr_value))
                print(f"      ❌ MANCANTE - {response[:80]}...")
                
        except Exception as e:
            print(f"      ⚠️ Errore verifica: {e}")
            # In caso di errore, consideriamo l'attributo mancante (conservativo)
            missing_attributes.append((attr_key, attr_value))
    
    # Calcola accuratezza
    total_attrs = len(attrs_to_check)
    accuracy = len(present_attributes) / total_attrs if total_attrs > 0 else 0.0
    
    print(f"\n   📊 Risultato Verifica:")
    print(f"      Presenti: {len(present_attributes)}/{total_attrs}")
    print(f"      Mancanti: {len(missing_attributes)}/{total_attrs}")
    print(f"      Accuracy: {accuracy:.1%}")
    
    return {
        "missing": missing_attributes,
        "present": present_attributes,
        "accuracy": accuracy,
        "details": details
    }, model


def verify_description_fallback(gen_img, fingerprints, model):
    """Fallback per verifica basata su descrizione testuale"""
    description = fingerprints.get("description", "")
    
    msgs = [{
        "role": "user",
        "content": f"Does this image match this description: '{description}'? Answer Yes/No and explain."
    }]
    
    with torch.no_grad():
        result = model.model_interface.model.chat(
            image=gen_img,
            msgs=msgs,
            tokenizer=model.model_interface.tokenizer
        )
        response = result[-1] if isinstance(result, tuple) else str(result)
    
    is_match = "yes" in response.lower()
    
    return {
        "missing": [] if is_match else [("description", description)],
        "present": ["description"] if is_match else [],
        "accuracy": 1.0 if is_match else 0.0,
        "details": {"description": response}
    }, model


# Mantieni anche la versione semplice per retrocompatibilità
def verify_generation(generated_image_path, reference_image_path, fingerprints_dict, model=None):
    """Wrapper di compatibilità - usa verifica granulare"""
    verification, model = verify_generation_detailed(generated_image_path, fingerprints_dict, model)
    score = verification["accuracy"]
    return score, model


if __name__ == "__main__":
    # Test
    test_fp = {
        "brand": "Nike",
        "color": "red and white",
        "product_type": "sneaker"
    }
    
    test_img = "output/test.png"
    if os.path.exists(test_img):
        result, _ = verify_generation_detailed(test_img, test_fp)
        print("\n" + "="*60)
        print("DETTAGLI VERIFICA:")
        for attr, response in result["details"].items():
            print(f"\n{attr}:")
            print(f"  {response}")