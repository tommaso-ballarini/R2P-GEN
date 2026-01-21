import os
import sys
import glob
import json
import argparse
from tqdm import tqdm

# --- 1. CONFIGURATION ---

# Paths (Relative to 'pipeline2/')
# Assumes images are in: pipeline2/data/perva_data/<folder>/<image>
SOURCE_DATA_DIR = "data/perva-data" 
OUTPUT_JSON_PATH = "database/database_perva.json"

# Model Configuration
DEVICE = "cuda"
MODEL_PATH = "openbmb/MiniCPM-o-2_6"

# Run Settings
DEBUG_MODE = True         # Set to False to process the entire dataset
DEBUG_LIMIT = 5           # Number of images to process in debug mode
USE_CLIP_CATEGORY = True  # Flag: True = Let R2P detect category via CLIP; False = Pass None/Generic

# --- 2. SDXL PROMPT TEMPLATES ---
# You can switch between these templates by changing the index in 'SELECTED_PROMPT_INDEX'

# --- 2. SDXL PROMPT TEMPLATES ---
# Chiediamo all'LLM SOLO la descrizione fisica. Lo stile lo aggiungiamo noi via codice.
SYSTEM_PROMPT_TRANSLATOR = """
You are an expert Visual Describer.
You will receive a technical sheet of a product.
Convert it into a single, comma-separated description of the physical object.

RULES:
1. Start with the Subject (Category).
2. Describe visuals: material, color, shape.
3. Include specific details: visible text, patterns, flaws.
4. DO NOT add style tags (like '8k', 'photo'). Just describe the object.
5. Output ONLY the string.
"""

# Stile fisso che verrà incollato a ogni prompt automaticamente
HARDCODED_STYLE = " Studio lighting, 8k, sharp focus, hyperrealistic, texture details, highly detailed"


# --- 3. SETUP & IMPORTS (FIXED) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# Definiamo le cartelle chiave
core_dir = os.path.join(current_dir, "r2p_core")
utils_dir = os.path.join(core_dir, "utils")
models_dir = os.path.join(core_dir, "models")
database_dir = os.path.join(core_dir, "database")
eval_dir = os.path.join(core_dir, "evaluators")

# Aggiungiamo TUTTO al path di sistema
for path in [core_dir, utils_dir, models_dir, database_dir, eval_dir]:
    if path not in sys.path:
        sys.path.append(path)

try:
    # Ora possiamo importare direttamente dai moduli interni
    from database.mini_cpm_info import MiniCPMDescription
    
    # Classe Mock per compatibilità R2P (necessaria)
    class MockArgs:
        def __init__(self):
            self.user_defined = False
            self.template_based = True
            
except ImportError as e:
    print(f"❌ Error importing R2P modules: {e}")
    print("Ensure 'r2p_core' contains the 'src' files from the original repo.")
    sys.exit(1)

# --- 4. HELPER FUNCTIONS ---

def get_image_files(root_dir):
    """
    Trova immagini jpg/png ricorsivamente.
    Corregge automaticamente errori nel nome cartella (es. _ vs -).
    """
    # Debug del percorso assoluto
    abs_path = os.path.abspath(root_dir)
    print(f"   🔍 Looking in: {abs_path}")
    
    target_dir = root_dir
    
    # Logica di correzione percorso
    if not os.path.exists(root_dir):
        print(f"   ⚠️ Path '{root_dir}' not found.")
        # Prova varianti comuni
        if "_" in root_dir and os.path.exists(root_dir.replace("_", "-")):
            target_dir = root_dir.replace("_", "-")
            print(f"   ✅ Found '{target_dir}' instead. Switching.")
        elif "-" in root_dir and os.path.exists(root_dir.replace("-", "_")):
            target_dir = root_dir.replace("-", "_")
            print(f"   ✅ Found '{target_dir}' instead. Switching.")
        # Prova cartella genitore (comune in certi ambienti)
        elif os.path.exists(os.path.join("..", root_dir)):
            target_dir = os.path.join("..", root_dir)
            print(f"   ✅ Found in parent directory. Switching.")
        else:
            # Fallimento
            return []

    # Scansione con os.walk (più sicuro di glob)
    valid_exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    files = []
    
    for root, _, filenames in os.walk(target_dir):
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext in valid_exts:
                files.append(os.path.join(root, filename))
                
    return sorted(files)

def collect_features_safely(info_dict):
    """
    IL COLLEZIONISTA (The Collector):
    Aggrega dinamicamente tutti i campi disponibili in una 'Scheda Tecnica' testuale.
    Gestisce campi mancanti o vuoti in modo sicuro.
    """
    lines = []
    
    # Mappa dei campi R2P -> Etichetta Leggibile per LLM
    field_map = [
        ('category', 'Category'),
        ('general', 'Base Description'),
        ('shape', 'Shape'),
        ('material', 'Material'),
        ('color', 'Color'),
        ('pattern', 'Pattern'),
        ('brand/text', 'Visible Text'),
        ('distinct features', 'Distinctive Flaws/Details')
    ]

    # Blacklist: se il valore contiene queste parole, scartiamo l'intera riga.
    negative_triggers = ["no visible", "none", "n/a", "not readable", "unknown", "no brand", "no text"]

    for key, label in field_map:
        val = info_dict.get(key)
        
        if val and isinstance(val, str):
            clean_val = val.strip()
            val_lower = clean_val.lower()
            
            # Controllo lunghezza minima
            if len(clean_val) < 2: continue
            
            # FILTRO ANTI-NEGAZIONI: Se contiene parole vietate, SALTA.
            if any(trigger in val_lower for trigger in negative_triggers):
                continue
            
            lines.append(f"{label}: {clean_val}")

    return "\n".join(lines)


def generate_sdxl_prompt(extractor, full_info_dict):
    """
    Genera il prompt SDXL usando la Scheda Tecnica aggregata invece che solo i fingerprints.
    """
    # 1. Costruisci l'input ricco (Il Collezionista)
    features_text = collect_features_safely(full_info_dict)
    
    # Fallback se non abbiamo trovato nulla
    if not features_text:
        return f"A photorealistic image of a {full_info_dict.get('category', 'object')}, studio lighting."

    query = f"{SYSTEM_PROMPT_TRANSLATOR}\n\n--- INPUT TECHNICAL SHEET ---\n{features_text}"
    
    try:
        msgs = [{"role": "user", "content": query}]
        
        # Chiamata al modello
        res = extractor.model.chat(msgs=msgs, tokenizer=extractor.tokenizer)
        
        # Gestione Tuple
        if isinstance(res, (tuple, list)):
            description = res[0]
        else:
            description = res
            
        # --- MODIFICA QUI: Pulizia e Aggiunta Stile ---
        # 1. Rimuove virgolette accidentali dall'LLM
        clean_desc = description.strip().strip('"').strip("'")
        
        # 2. Incolla lo stile fisso in coda
        final_prompt = f"{clean_desc}{HARDCODED_STYLE}"
        
        return final_prompt
                   
    except Exception as e:
        print(f"⚠️ Warning: Failed to generate SDXL prompt: {e}")
        return features_text

# --- 5. MAIN SCRIPT ---

def main():
    print(f"🚀 Starting Build Database Script")
    print(f"📂 Source: {SOURCE_DATA_DIR}")
    print(f"💾 Output: {OUTPUT_JSON_PATH}")
    print(f"🔧 Device: {DEVICE}")
    print(f"🧪 Debug Mode: {DEBUG_MODE}")
    
    # 1. Initialize R2P Extractor
    print("\n[1/4] Loading MiniCPM Model (this may take a moment)...")
    extractor = MiniCPMDescription(model_path=MODEL_PATH, device=DEVICE)
    mock_args = MockArgs()
    
    # 2. Find Images
    print("\n[2/4] Scanning for images...")
    
    # Troviamo tutte le immagini
    all_images = get_image_files(SOURCE_DATA_DIR)
    
    if not all_images:
        print(f"❌ No images found in {SOURCE_DATA_DIR}. Check the path.")
        return

    print(f"Found {len(all_images)} total images.")
    
    target_images = []
    
    if DEBUG_MODE:
        print(f"⚠️ DEBUG MODE ACTIVE: Selecting diverse images...")
        # LOGICA MIGLIORATA: Raggruppa per cartella padre per garantire varietà
        images_by_folder = {}
        for img_path in all_images:
            folder = os.path.dirname(img_path)
            if folder not in images_by_folder:
                images_by_folder[folder] = []
            images_by_folder[folder].append(img_path)
        
        # Prendi 1 immagine da ogni cartella trovata, fino a raggiungere DEBUG_LIMIT
        folders = sorted(list(images_by_folder.keys()))
        
        # Cicla sulle cartelle e prendi la prima immagine di ognuna
        count = 0
        for folder in folders:
            if count >= DEBUG_LIMIT:
                break
            # Prende la prima immagine della cartella corrente
            target_images.append(images_by_folder[folder][0])
            count += 1
            
        print(f"   Selected {len(target_images)} images from {len(target_images)} different categories.")
    else:
        target_images = all_images

    # 3. Process Loop
    print("\n[3/4] Extracting Fingerprints & Generating Prompts...")
    
    # Structure strictly following R2P format
    database_data = {
        "concept_dict": {},
        "path_to_concept": {}
    }
    
    success_count = 0
    
    for img_path in tqdm(target_images):
        try:

            # --- MODIFICA A: ID Univoco ---
            filename = os.path.basename(img_path)
            file_stem = os.path.splitext(filename)[0] # Es. "1"
            parent_folder = os.path.basename(os.path.dirname(img_path)) # Es. "dbi"
            
            # Creiamo un ID univoco: "prodotto_numero" (es. "dbi_1")
            concept_id = f"{parent_folder}_{file_stem}"
            
            # --- A. R2P EXTRACTION ---
            # category=None forces R2P to use CLIP to detect category (if USE_CLIP_CATEGORY is True)
            # If you wanted to force a category, you could extract it from the folder name here.
            cat_arg = None if USE_CLIP_CATEGORY else "object"
            
            # generate_caption returns a JSON string
            json_str = extractor.generate_caption(
                image_file=img_path,    # <--- ERA image_path
                cat=cat_arg,            # <--- ERA category
                concept_identifier=concept_id, 
                args=mock_args
            )
            
            # Parse the JSON string
            # R2P sometimes returns markdown code blocks, strip them if needed
            json_str = json_str.replace('```json', '').replace('```', '').strip()
            item_info = json.loads(json_str)
            
            # --- B. SDXL PROMPT TRANSLATION (COLLECTOR EDITION) ---
            # Passiamo tutto il dizionario item_info, il Collezionista sceglierà cosa usare
            sdxl_prompt = generate_sdxl_prompt(extractor, item_info)
            
            # --- C. SAVE TO DB STRUCTURE ---
            # Inject sdxl_prompt into the info dictionary
            item_info["sdxl_prompt"] = sdxl_prompt
            
            # Key format often used by R2P: <concept_id>
            key = f"<{concept_id}>"
            
            database_data["concept_dict"][key] = {
                "name": concept_id,
                "image": [img_path], # List format as per R2P
                "info": item_info,
                "category": item_info.get("category", "unknown")
            }
            
            database_data["path_to_concept"][img_path] = key
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ Error processing {os.path.basename(img_path)}: {e}")
            continue

    # 4. Save to Disk
    print(f"\n[4/4] Saving database to {OUTPUT_JSON_PATH}...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(database_data, f, indent=4)
        
    print(f"✅ Done! Processed {success_count}/{len(target_images)} images.")
    print(f"   Database saved at: {os.path.abspath(OUTPUT_JSON_PATH)}")

if __name__ == "__main__":
    main()