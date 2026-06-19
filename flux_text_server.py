"""
flux_text_server.py
===================
Server FastAPI per FluxText — correzione testo in immagine.
Gira su GPU 2, porta 8767.
"""

import os
import io
import sys
import base64
import yaml
import torch
import argparse
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn

FLUXTEXT_CODE = "/leonardo_work/IscrC_MUSE/tballari/models_cache/FluxText/code"
sys.path.insert(0, FLUXTEXT_CODE)

from src.flux.condition import Condition
from src.flux.generate_fill import generate_fill
from src.train.model import OminiModelFIll
from safetensors.torch import load_file

# ── Configurazione ───────────────────────────────────────────────────────────
FLUXTEXT_MODEL_DIR  = "/leonardo_work/IscrC_MUSE/tballari/models_cache/FluxText"
CONFIG_PATH         = f"{FLUXTEXT_MODEL_DIR}/model_multisize/config.yaml"
LORA_PATH           = f"{FLUXTEXT_MODEL_DIR}/model_multisize/pytorch_lora_weights.safetensors"
FONT_PATH           = f"{FLUXTEXT_CODE}/font/Arial_Unicode.ttf"
STEPS               = 28

# ── Caricamento modello ──────────────────────────────────────────────────────
print("🚀 Caricamento FluxText in VRAM (GPU 2)...")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

training_config = config["train"]
trainable_model = OminiModelFIll(
    flux_pipe_id=config["flux_path"],
    lora_config=training_config["lora_config"],
    device="cuda:0",  # CUDA_VISIBLE_DEVICES=2 nel job → appare come cuda:0
    dtype=getattr(torch, config["dtype"]),
    optimizer_config=training_config["optimizer"],
    model_config=config.get("model", {}),
    gradient_checkpointing=False,
    byt5_encoder_config=None,
)

state_dict = load_file(LORA_PATH)
state_dict_new = {
    k.replace('lora_A', 'lora_A.default')
     .replace('lora_B', 'lora_B.default')
     .replace('transformer.', ''): v
    for k, v in state_dict.items()
}
trainable_model.transformer.load_state_dict(state_dict_new, strict=False)
pipe = trainable_model.flux_pipe

font = ImageFont.truetype(FONT_PATH, size=60)
generator = torch.Generator(device="cuda:0")

print("✅ FluxText pronto.")

app = FastAPI()


# ── Schemi ───────────────────────────────────────────────────────────────────

class TextFixRequest(BaseModel):
    source_image_b64: str       # immagine generata con testo sbagliato
    mask_b64: str               # maschera binaria zona testo (RGB, bianco=testo)
    texts: list[str]            # testi da renderizzare (già puliti)
    prompt: str                 # prompt testuale completo
    seed: int = 42
    height: Optional[int] = None
    width: Optional[int] = None


class TextFixResponse(BaseModel):
    image_b64: str
    error: str


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_glyph(texts: list[str], mask_array: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Genera il glyph image: testo bianco su sfondo nero,
    posizionato nell'area mascherata.
    Restituisce array float32 normalizzato (255 - glyph) / 255.
    """
    # Trova bounding box della mask
    rows = np.any(mask_array > 128, axis=1)
    cols = np.any(mask_array > 128, axis=0)

    if not rows.any():
        # Mask vuota — fallback centro immagine
        y1, y2 = height // 4, 3 * height // 4
        x1, x2 = width // 4, 3 * width // 4
    else:
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

    box_w = max(int(x2 - x1), 10)
    box_h = max(int(y2 - y1), 10)

    # Calcola font size proporzionale alla bounding box
    text_content = "\n".join(texts)
    n_lines = len(texts)
    font_size = max(12, min(box_h // max(n_lines, 1), box_w // max(len(max(texts, key=len)), 1)))
    try:
        dyn_font = ImageFont.truetype(FONT_PATH, size=font_size)
    except Exception:
        dyn_font = font

    # Disegna testo bianco su sfondo nero
    glyph_img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(glyph_img)

    # Centra il testo nella bounding box
    y_cursor = y1
    line_h = box_h // max(n_lines, 1)
    for line in texts:
        bbox = draw.textbbox((0, 0), line, font=dyn_font)
        text_w = bbox[2] - bbox[0]
        x_centered = x1 + (box_w - text_w) // 2
        draw.text((x_centered, y_cursor), line, font=dyn_font, fill=(255, 255, 255))
        y_cursor += line_h

    glyph_array = np.array(glyph_img)
    # FluxText vuole (255 - glyph) / 255
    return (255 - glyph_array) / 255.0


# ── Endpoint ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/fix_text", response_model=TextFixResponse)
def fix_text(req: TextFixRequest):
    try:
        # Decode source image
        src_bytes = base64.b64decode(req.source_image_b64)
        source_img = Image.open(io.BytesIO(src_bytes)).convert("RGB")

        # Decode mask
        mask_bytes = base64.b64decode(req.mask_b64)
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("RGB")

        # Dimensioni target
        tgt_w = req.width  or source_img.width
        tgt_h = req.height or source_img.height

        # Resize a multipli di 16 (requisito FLUX)
        tgt_w = (tgt_w // 16) * 16
        tgt_h = (tgt_h // 16) * 16

        source_img = source_img.resize((tgt_w, tgt_h))
        mask_img   = mask_img.resize((tgt_w, tgt_h))

        mask_array = np.array(mask_img)[:, :, 0]  # canale R come grayscale

        # Hint: mask normalizzata
        hint_array = np.array(mask_img) / 255.0

        # Glyph: testo renderizzato nell'area della mask
        glyph_array = _build_glyph(req.texts, mask_array, tgt_w, tgt_h)

        # Assembla condition nel formato FluxText
        condition_img = [glyph_array, hint_array, source_img]
        condition = Condition(
            condition_type='word_fill',
            condition=condition_img,
            position_delta=[0, 0],
        )

        generator.manual_seed(req.seed)
        res = generate_fill(
            pipe,
            prompt=req.prompt,
            conditions=[condition],
            height=tgt_h,
            width=tgt_w,
            generator=generator,
            num_inference_steps=STEPS,
            model_config=config.get("model", {}),
            default_lora=True,
        )

        out_img = res.images[0]
        buf = io.BytesIO()
        out_img.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        return TextFixResponse(image_b64=img_b64, error="")

    except Exception as e:
        return TextFixResponse(image_b64="", error=str(e))


# ── Avvio ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    print(f"🌐 Avvio FluxText server sulla porta {args.port}...")
    uvicorn.run(app, host="127.0.0.1", port=args.port) 