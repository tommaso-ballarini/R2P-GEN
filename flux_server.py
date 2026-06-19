"""
flux_server.py
==============
Server FastAPI minimale che espone FLUX come endpoint HTTP.
Lanciato da recovery_pipeline.sh in background con flux_test_work.
Gira sulla porta 8766.
Processa sempre un'immagine alla volta per evitare OOM.
"""

import os
import io
import base64
import torch
import argparse  
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn

# ── Patch Diffusers ──────────────────────────────────────────────────────────
import diffusers.models.attention_processor
import torch.nn.functional as F

_original_sdpa = F.scaled_dot_product_attention
def _stripped_sdpa(*args, **kwargs):
    kwargs.pop('enable_gqa', None)
    return _original_sdpa(*args, **kwargs)
diffusers.models.attention_processor.F.scaled_dot_product_attention = _stripped_sdpa

from diffusers import DiffusionPipeline

# ── Configurazione ───────────────────────────────────────────────────────────
FLUX_MODEL_DIR = "/leonardo_work/IscrC_MUSE/tballari/models_cache/FLUX.2-klein-9B"
STEPS          = 4

# ── Caricamento modello (una volta sola all'avvio) ───────────────────────────
print("🚀 Caricamento FLUX in VRAM...")
pipe = DiffusionPipeline.from_pretrained(
    FLUX_MODEL_DIR,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    trust_remote_code=True,
).to("cuda:0")
print("✅ FLUX pronto.")

app = FastAPI()


# ── Schemi request/response ──────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompts: list[str]
    seeds: list[int]
    source_image_b64: Optional[list[str]] = None


class GenerateResponse(BaseModel):
    images_b64: list[str]
    errors: list[str]


# ── Logica generazione singola immagine ──────────────────────────────────────

def _generate_one(prompt: str, seed: int, source_b64: Optional[str]) -> tuple[str, str]:
    """
    Genera una singola immagine. Ritorna (image_b64, error_string).
    error_string è "" se OK, messaggio di errore altrimenti.
    """
    try:
        generator = torch.Generator(device="cuda:0").manual_seed(seed)
        kwargs = {
            "prompt": [prompt],
            "num_inference_steps": STEPS,
            "generator": [generator],
        }
        if source_b64 is not None:
            img_bytes = base64.b64decode(source_b64)
            source_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            kwargs["image"] = [source_img]

        output = pipe(**kwargs).images[0]

        buf = io.BytesIO()
        output.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode(), ""

    except Exception as e:
        return "", f"FLUX Crash: {e}"


# ── Endpoint ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """
    Accetta batch di N immagini ma le processa una alla volta
    per evitare OOM. Il chiamante non deve cambiare nulla.
    """
    images_b64 = []
    errors = []

    for i in range(len(req.prompts)):
        source_b64 = req.source_image_b64[i] if req.source_image_b64 else None
        img_b64, error = _generate_one(req.prompts[i], req.seeds[i], source_b64)
        images_b64.append(img_b64)
        errors.append(error)

    return GenerateResponse(images_b64=images_b64, errors=errors)


# ── Avvio ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server FLUX via FastAPI")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    print(f"🌐 Avvio Uvicorn sulla porta dinamica: {args.port}...")
    uvicorn.run(app, host="127.0.0.1", port=args.port)