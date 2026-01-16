# import sys
# import os
# import torch
# from PIL import Image

# # Setup Path per importare i moduli R2P copiati in r2p_core
# current_dir = os.path.dirname(os.path.abspath(__file__))
# r2p_path = os.path.join(current_dir, 'r2p_core')
# sys.path.append(r2p_path)

# # Importa dai moduli originali R2P
# try:
#     from models.mini_cpm_reasoning import MiniCPMReasoning
#     # Usiamo la funzione specifica per Retail (PerVA) come da documentazione
#     from database.mini_cpm_info import MiniCPMDescription
# except ImportError as e:
#     print(f"❌ Errore Import R2P: {e}")
#     print("Assicurati di aver copiato le cartelle 'models' e 'database' dentro 'r2p_core'")
#     sys.exit(1)

# def extract_fingerprints(image_path, model=None):
#     """
#     Estrae i tratti distintivi.
#     Se 'model' è passato, usa quello già caricato (risparmia tempo nel full loop).
#     """
#     print(f"\n🔍 [STEP 1] Estrazione Fingerprints: {os.path.basename(image_path)}")
    
#     # 1. Carica Modello (se non fornito)
#     if model is None:
#         print("   -> Caricamento VLM (MiniCPM)...")
#         # Sul cluster userà la GPU automaticamente e scaricherà il modello
#         model = MiniCPMReasoning() 

#     # 2. Prepara Immagine e Prompt
#     image = Image.open(image_path).convert('RGB')
    
#     # Usa il prompt specifico per oggetti Retail (PerVA)
#     # Questo genera il template JSON che forza il modello a cercare dettagli
#     msgs = MiniCPMDescription()
    
#     # 3. Inferenza
#     try:
#         response = model.model.chat(
#             image=image,
#             msgs=msgs,
#             context=None
#         )
#         print(f"   ✅ Fingerprints estratte: {response[:100]}...") # Anteprima
#         return response, model
#     except Exception as e:
#         print(f"   ❌ Errore durante l'estrazione: {e}")
#         return None, model

# if __name__ == "__main__":
#     # Test Standalone
#     test_img = "data/perva_test/1.jpg" # Assicurati che esista
#     if os.path.exists(test_img):
#         res, _ = extract_fingerprints(test_img)
#         print("Risultato completo:", res)
#     else:
#         print(f"File {test_img} non trovato.")



import sys
import os
import torch
import gc
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
r2p_path = os.path.join(current_dir, 'r2p_core')
sys.path.append(r2p_path)

try:
    from models.mini_cpm_reasoning import MiniCPMReasoning
    # We do not need MiniCPMDescription if we use direct prompting
except ImportError as e:
    print(f"❌ Errore Import R2P: {e}")
    sys.exit(1)

def print_memory_stats(label=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved
        print(f"   📊 {label}")
        print(f"      Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Free: {free:.2f}GB")

def extract_fingerprints(image_path, model=None):
    print(f"\n🔍 [STEP 1] Estrazione Fingerprints: {os.path.basename(image_path)}")
    print_memory_stats("PRIMA del caricamento modello")
    
    if model is None:
        torch.cuda.empty_cache()
        gc.collect()
        
        print("   -> Caricamento VLM (MiniCPM-o-2_6)...")
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        try:
            model = MiniCPMReasoning(
                model_path="openbmb/MiniCPM-o-2_6",
                device="cuda",
                torch_dtype=torch.float16,
                attn_implementation="sdpa",
                seed=42
            )
            print("   ✅ Modello caricato!")
            print_memory_stats("DOPO caricamento modello")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("   ❌ OOM durante caricamento!")
                torch.cuda.empty_cache()
                model = MiniCPMReasoning(
                    model_path="openbmb/MiniCPM-o-2_6",
                    device="auto",
                    torch_dtype=torch.float16,
                    attn_implementation="sdpa",
                    seed=42
                )
            else:
                raise e
    
    # Prepara immagine
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    max_dim = 896
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        print(f"   -> Immagine ridimensionata: {original_size} → {image.size}")
    
    # Prompt per estrazione attributi
    prompt = """Analyze this retail product image and extract detailed visual attributes in JSON format:
{
  "brand": "detected brand name or logo",
  "product_type": "specific product category",
  "color": "dominant colors",
  "material": "visible materials/textures",
  "shape": "geometric characteristics",
  "size_appearance": "relative size indicators",
  "packaging": "packaging details if visible",
  "distinctive_features": "unique visual elements"
}"""
    
    msgs = [{'role': 'user', 'content': prompt}]
    
    print_memory_stats("PRIMA dell'inferenza")
    
    try:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                # CHIAMATA DIRETTA al modello HF per evitare il wrapper buggy
                result = model.model_interface.model.chat(
                    image=image,  # Passa l'immagine direttamente
                    msgs=msgs, 
                    tokenizer=model.model_interface.tokenizer
                )
                
                # Debug
                print(f"   🔍 Debug - tipo: {type(result)}, len: {len(result) if hasattr(result, '__len__') else 'N/A'}")
                
                # Estrai la risposta
                if isinstance(result, tuple):
                    # Prendi l'ultimo elemento che di solito è la risposta
                    answer = result[-1] if result else ""
                else:
                    answer = str(result)
                
        print(f"   ✅ Fingerprints estratte ({len(answer)} chars)")
        print(f"   Preview: {answer[:200]}...")
        print_memory_stats("DOPO inferenza")
        
        return answer, model
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"   ❌ OOM durante inferenza!")
        print_memory_stats("Al momento dell'errore")
        
        print("   -> Retry con risoluzione dimezzata...")
        torch.cuda.empty_cache()
        
        smaller_dim = max_dim // 2
        ratio = smaller_dim / max(original_size)
        new_size = tuple(int(dim * ratio) for dim in original_size)
        image = Image.open(image_path).convert('RGB').resize(new_size, Image.Resampling.LANCZOS)
        print(f"   -> Nuova dimensione: {image.size}")
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                result = model.model_interface.model.chat(
                    image=image,
                    msgs=msgs, 
                    tokenizer=model.model_interface.tokenizer
                )
                answer = result[-1] if isinstance(result, tuple) else str(result)
        
        return answer, model
        
    except Exception as e:
        print(f"   ❌ Errore generico: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None, model
    

    
if __name__ == "__main__":
    print(f"🚀 Test Extraction su L4")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    test_img = "data/perva_test/1.jpg"
    if os.path.exists(test_img):
        res, model = extract_fingerprints(test_img)
        if res:
            print("\n" + "="*60)
            print("📄 RISULTATO COMPLETO:")
            print("="*60)
            print(res)
            print("="*60)
    else:
        print(f"❌ File {test_img} non trovato.")