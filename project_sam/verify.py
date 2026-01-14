import sys
import os
import torch
from PIL import Image

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'r2p_core'))

from models.mini_cpm_reasoning import MiniCPMReasoning
# Se R2P ha questa funzione specifica per il confronto, la usiamo
from models.prompt_generator import get_image2image_plus_text_comparison_prompt

def verify_generation(generated_image_path, reference_image_path, fingerprints, model=None):
    print(f"\n⚖️ [STEP 3] Verifica/Reasoning Loop")

    if model is None:
        print("   -> Ricaricamento VLM per la verifica...")
        model = MiniCPMReasoning()
    
    gen_img = Image.open(generated_image_path).convert('RGB')
    
    # 1. Costruzione Prompt del Giudice
    # Creiamo un prompt che chiede esplicitamente se i fingerprint sono presenti
    # Nota: Possiamo usare 'get_image2image_plus_text_comparison_prompt' se vogliamo
    # confrontare Ref vs Gen, oppure un prompt diretto sull'immagine generata.
    # Per semplicità e robustezza iniziale, usiamo un prompt diretto:
    
    msgs = [
        {"role": "user", "content": f"Analyze this image. Does it clearly contain the following attributes: {fingerprints}? Answer 'Yes' only if all attributes are present, otherwise 'No'. Explain briefly."}
    ]
    
    # 2. Inferenza
    try:
        # Passiamo l'immagine GENERATA al modello
        res = model.model.chat(
            image=gen_img,
            msgs=msgs,
            context=None
        )
        print(f"   -> Risposta del Giudice: {res}")
        
        # 3. Calcolo Score Semplificato
        if "Yes" in res or "yes" in res:
            score = 1.0
        else:
            score = 0.0
            
        print(f"   ✅ Confidence Score: {score}")
        return score, model

    except Exception as e:
        print(f"   ❌ Errore nel Giudice: {e}")
        return 0.0, model

if __name__ == "__main__":
    # Test Standalone
    print("Test verifica...")