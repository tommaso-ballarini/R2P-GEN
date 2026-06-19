"""
Modulo di Refinement per R2P-GEN (FLUX Edition).
Contiene le logiche per l'interazione con i server API (VLM e FLUX) 
per tentare il recupero delle immagini fallite.
"""
import os
import io
import json
import base64
import requests
from PIL import Image


def qwen3_rewrite_prompt(
    reasoner,
    original_prompt: str,
    missing_details: list,
    attempt: int,
    failed_image_path: str = None,
    attempts_history: list = None,  
) -> str:
    """
    Riscrive il prompt usando Qwen3-VL in modalità multimodale se disponibile
    l'immagine del tentativo fallito, altrimenti testo-only.
    Integra la history dei tentativi precedenti per evitare ripetizioni.
    """
    from PIL import Image

    missing_str = ", ".join(missing_details)

   
    history_context = ""
    if attempts_history:
        history_context = "\nPrevious attempts summary:\n"
        for h in attempts_history:
            status = "✓ improved" if h["improved"] else "✗ no improvement"
            history_context += (
                f"  - Attempt {h['attempt']}: "
                f"missing {h['missing_before']} → after: {h['missing_after']} [{status}]\n"
            )
        history_context += "\n"

    # --- Costruzione messaggio visuale (se abbiamo l'immagine fallita) ---
    if failed_image_path and os.path.exists(failed_image_path):
        try:
            failed_image = Image.open(failed_image_path).convert("RGB")

            if attempt == 1:
                instruction = (
                    f"This image was generated but failed verification. "
                    f"Look at the image and identify why these attributes are missing or incorrect: {missing_str}.\n"
                    f"Rewrite the prompt placing the missing attributes at the BEGINNING of the prompt, "
                    f"preceded by descriptive adjectives: 'highly detailed', 'prominent', 'clearly visible'.\n"
                    f"Keep scene, object category, and framing EXACTLY the same.\n"
                    f"{history_context}"
                    f"Original prompt: {original_prompt}\n"
                    f'Respond ONLY with JSON: {{"rewritten_prompt": "..."}}'
                )
            elif attempt == 2:
                instruction = (
                    f"This image still fails to show: {missing_str}.\n"
                    f"Look at the image carefully. Rewrite the prompt using explicit weight syntax "
                    f"'(attribute:1.4)' for each missing attribute, and move them to the very start.\n"
                    f"Example: '(ZODIAC logo:1.4), (damask pattern:1.4), on wooden table...'\n"
                    f"Keep scene context and background unchanged.\n"
                    f"{history_context}"
                    f"Original prompt: {original_prompt}\n"
                    f'Respond ONLY with JSON: {{"rewritten_prompt": "..."}}'
                )
            else:
                instruction = (
                    f"This image has REPEATEDLY failed to show: {missing_str}.\n"
                    f"Look at the image. The standard prompt approach has failed twice.\n"
                    f"Rewrite the prompt making it almost entirely focused on the missing attributes: "
                    f"use '(attribute:1.5)' weighting, repeat the critical attribute twice in different positions, "
                    f"reduce all other scene description to minimum.\n"
                    f"Example: '(ZODIAC logo:1.5) prominently displayed, beige damask bag, (ZODIAC logo:1.5) clearly visible'\n"
                    f"Keep only essential object identity, drop style/background filler.\n"
                    f"{history_context}"
                    f"Original prompt: {original_prompt}\n"
                    f'Respond ONLY with JSON: {{"rewritten_prompt": "..."}}'
                )
                
            msgs = reasoner.adapter.format_text_options_msgs(failed_image, instruction)
            _, gen_text = reasoner.model_interface.chat(msgs)

        except Exception as e:
            print(f"  [WARN] Fallback a testo-only per errore immagine: {e}")
            failed_image_path = None  # trigger fallback sotto

    # --- Fallback testo-only ---
    if not failed_image_path or not os.path.exists(failed_image_path):
        sys_msg = (
            "You are an expert AI prompt engineer for image diffusion models. "
            "Your task is to rewrite the prompt to recover missing physical details of an object. "
            'Output ONLY valid JSON: {"rewritten_prompt": "..."}'
        )
        if attempt == 1:
            user_msg = (
                f"The image missed these critical details: {missing_str}.\n"
                f"Rewrite placing missing attributes at the START with adjectives "
                f"'highly detailed', 'prominent', 'clearly visible'.\n"
                f"Keep scene, object category, and framing EXACTLY the same.\n"
                f"{history_context}"
                f"Original prompt: {original_prompt}"
            )
        elif attempt == 2:
            user_msg = (
                f"The image still fails to show: {missing_str}.\n"
                f"Use explicit weight syntax '(attribute:1.4)' for each missing attribute, "
                f"move them to the very start of the prompt.\n"
                f"Example: '(ZODIAC logo:1.4), (damask pattern:1.4), on wooden table...'\n"
                f"Keep scene context and background unchanged.\n"
                f"{history_context}"
                f"Original prompt: {original_prompt}"
            )
        else:
            user_msg = (
                f"The image has REPEATEDLY failed to show: {missing_str}.\n"
                f"Make the prompt almost entirely focused on missing attributes: "
                f"use '(attribute:1.5)', repeat the critical attribute twice, "
                f"reduce all other description to minimum.\n"
                f"Example: '(ZODIAC logo:1.5) prominently displayed, beige damask bag, (ZODIAC logo:1.5) clearly visible'\n"
                f"{history_context}"
                f"Original prompt: {original_prompt}"
            )


        msgs = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]
        _, gen_text = reasoner.model_interface.chat(msgs)

    try:
        clean_text = gen_text.strip().removeprefix('```json').removesuffix('```').strip()
        parsed = json.loads(clean_text)
        return parsed.get('rewritten_prompt', original_prompt)
    except Exception as e:
        print(f"  [WARN] Fallito parsing JSON: {e}\nRaw: {gen_text}")
        return original_prompt


def generate_recovery_http(flux_url: str, source_image_path: str, prompt: str, seed: int, output_path: str, max_retries: int = 2) -> bool:
    """
    Invia l'immagine in Base64 e il nuovo prompt al server FLUX.
    Prima di ogni tentativo verifica che il server sia healthy.
    """
    if not os.path.exists(source_image_path):
        print(f"  ❌ Immagine sorgente non trovata: {source_image_path}")
        return False

    try:
        img = Image.open(source_image_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64_string = base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"  ❌ Errore lettura immagine sorgente: {e}")
        return False

    payload = {
        "prompts": [prompt],
        "seeds": [seed],
        "source_image_b64": [b64_string]
    }

    for retry in range(max_retries + 1):
        # Health-check prima di ogni tentativo
        try:
            health = requests.get(f"{flux_url}/health", timeout=5)
            if health.status_code != 200:
                raise ConnectionError("Server non healthy")
        except Exception as e:
            print(f"  ⚠️ FLUX non raggiungibile (retry {retry}/{max_retries}): {e}")
            if retry < max_retries:
                import time
                time.sleep(10)
                continue
            print("  ❌ FLUX irraggiungibile dopo tutti i retry.")
            return False

        # Chiamata generazione
        try:
            response = requests.post(f"{flux_url}/generate", json=payload, timeout=360)
            response.raise_for_status()
            result = response.json()

            error_msg = result.get('errors', [""])[0]
            if error_msg:
                print(f"  ❌ Errore interno server FLUX: {error_msg}")
                return False

            img_b64_response = result.get('images_b64', [""])[0]
            if not img_b64_response:
                print("  ❌ FLUX ha restituito un'immagine vuota.")
                return False

            img_bytes = base64.b64decode(img_b64_response)
            recovered_img = Image.open(io.BytesIO(img_bytes))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            recovered_img.save(output_path)
            return True

        except requests.Timeout:
            print(f"  ⚠️ Timeout FLUX (retry {retry}/{max_retries}).")
            if retry < max_retries:
                import time
                time.sleep(5)
                continue
            print("  ❌ FLUX timeout dopo tutti i retry.")
            return False

        except Exception as e:
            print(f"  ❌ Errore API FLUX: {e}")
            return False

    return False


def _generate_batch_http(
    flux_url: str,
    source_image_paths: list,
    prompts: list,
    seeds: list,
    output_paths: list,
    max_retries: int = 2,
) -> list[bool]:
    """
    Manda un batch di immagini+prompt al server FLUX in una singola chiamata HTTP.
    Ritorna una lista di bool (True=successo) per ogni elemento del batch.
    """
    results = [False] * len(prompts)

    # Prepara i base64
    sources_b64 = []
    for path in source_image_paths:
        try:
            img = Image.open(path).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            sources_b64.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
        except Exception as e:
            print(f"  ❌ Errore lettura {path}: {e}")
            sources_b64.append(None)

    # Se qualche immagine non è leggibile, fallback a stringa vuota
    # Il server gestirà l'errore per quell'elemento
    sources_b64_clean = [b or "" for b in sources_b64]

    payload = {
        "prompts": prompts,
        "seeds": seeds,
        "source_image_b64": sources_b64_clean,
    }

    for retry in range(max_retries + 1):
        try:
            health = requests.get(f"{flux_url}/health", timeout=5)
            if health.status_code != 200:
                raise ConnectionError("Server non healthy")
        except Exception as e:
            print(f"  ⚠️ FLUX non raggiungibile (retry {retry}/{max_retries}): {e}")
            if retry < max_retries:
                import time; time.sleep(10)
                continue
            return results

        try:
            response = requests.post(f"{flux_url}/generate", json=payload, timeout=360)
            response.raise_for_status()
            result = response.json()

            images_b64 = result.get("images_b64", [])
            errors = result.get("errors", [""] * len(prompts))

            for i, (img_b64, err) in enumerate(zip(images_b64, errors)):
                if err:
                    print(f"  ❌ Errore FLUX item {i}: {err}")
                    continue
                if not img_b64:
                    print(f"  ❌ FLUX item {i}: immagine vuota.")
                    continue
                try:
                    img_bytes = base64.b64decode(img_b64)
                    img = Image.open(io.BytesIO(img_bytes))
                    os.makedirs(os.path.dirname(output_paths[i]), exist_ok=True)
                    img.save(output_paths[i])
                    results[i] = True
                except Exception as e:
                    print(f"  ❌ Errore salvataggio item {i}: {e}")

            return results

        except requests.Timeout:
            print(f"  ⚠️ Timeout FLUX (retry {retry}/{max_retries}).")
            if retry < max_retries:
                import time; time.sleep(5)
                continue
            return results

        except Exception as e:
            print(f"  ❌ Errore API FLUX: {e}")
            return results

    return results