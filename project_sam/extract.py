import sys
import os
import torch
from PIL import Image

# Setup Path per importare i moduli R2P copiati in r2p_core
current_dir = os.path.dirname(os.path.abspath(__file__))
r2p_path = os.path.join(current_dir, 'r2p_core')
sys.path.append(r2p_path)

# Importa dai moduli originali R2P
try:
    from models.mini_cpm_reasoning import MiniCPMReasoning
    # Usiamo la funzione specifica per Retail (PerVA) come da documentazione
    from database.mini_cpm_info import MiniCPMDescription
except ImportError as e:
    print(f"❌ Errore Import R2P: {e}")
    print("Assicurati di aver copiato le cartelle 'models' e 'database' dentro 'r2p_core'")
    sys.exit(1)

def extract_fingerprints(image_path, model=None):
    """
    Estrae i tratti distintivi.
    Se 'model' è passato, usa quello già caricato (risparmia tempo nel full loop).
    """
    print(f"\n🔍 [STEP 1] Estrazione Fingerprints: {os.path.basename(image_path)}")
    
    # 1. Carica Modello (se non fornito)
    if model is None:
        print("   -> Caricamento VLM (MiniCPM)...")
        # Sul cluster userà la GPU automaticamente e scaricherà il modello
        model = MiniCPMReasoning() 

    # 2. Prepara Immagine e Prompt
    image = Image.open(image_path).convert('RGB')
    
    # Usa il prompt specifico per oggetti Retail (PerVA)
    # Questo genera il template JSON che forza il modello a cercare dettagli
    msgs = MiniCPMDescription()
    
    # 3. Inferenza
    try:
        response = model.model.chat(
            image=image,
            msgs=msgs,
            context=None
        )
        print(f"   ✅ Fingerprints estratte: {response[:100]}...") # Anteprima
        return response, model
    except Exception as e:
        print(f"   ❌ Errore durante l'estrazione: {e}")
        return None, model

if __name__ == "__main__":
    # Test Standalone
    test_img = "data/perva_test/1.jpg" # Assicurati che esista
    if os.path.exists(test_img):
        res, _ = extract_fingerprints(test_img)
        print("Risultato completo:", res)
    else:
        print(f"File {test_img} non trovato.")