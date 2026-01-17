## 📁 Quick File Reference


| File | Descrizione |
|------|-------------|
| **`full_loop.py`** | 🚀 **MAIN** - Orchestratore che esegue l'intera pipeline extraction → refinement → output |
| **`extract.py`** | 🔍 Estrae attributi visivi (brand, colore, materiale) da un'immagine usando MiniCPM VLM |
| **`generate.py`** | 🎨 Genera immagine candidata con SDXL + IP-Adapter a partire da fingerprints e reference image |
| **`verify.py`** | ⚖️ Verifica granularmente se ogni attributo target è presente nell'immagine generata |
| **`refine.py`** | 🔁 Loop iterativo che rigenera immagini migliorando i negative prompt fino a convergenza |

---

## ⚙️ File di Supporto

| File | Descrizione |
|------|-------------|
| **`config.py`** | ⚙️ Configurazione centralizzata (modelli, parametri generation, thresholds loop) |
| **`utils.py`** | 🛠️ Utility condivise (cleanup GPU memory, logging stats, path management) |

---

## 📂 Cartelle

| Cartella | Contenuto |
|----------|-----------|
| **`data/perva_test/`** | Dataset immagini prodotti retail per test e benchmark |
| **`output/`** | Immagini generate salvate (candidate_iter1.png, candidate_iter2.png, ...) |
| **`r2p_core/`** | Moduli originali R2P (models, database) copiati dalla repo ufficiale |

---

## 🔄 Flusso Esecuzione Tipico
```
full_loop.py
    ↓
extract.py → {fingerprints_dict}
    ↓
refine.py
    ├→ generate.py → candidate_iter1.png
    ├→ verify.py → score 65%
    ├→ generate.py → candidate_iter2.png (con negative prompt aggiornato)
    ├→ verify.py → score 88%
    └→ STOP (target raggiunto) → best_image
```
