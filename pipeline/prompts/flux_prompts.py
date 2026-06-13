def build_flux_prompt(attributes: dict, target_context: str) -> str:
    """
    Costruisce un prompt in linguaggio naturale ottimizzato per il T5 encoder di FLUX,
    unendo un dizionario di attributi fisici (Textual Anchoring) con un contesto target.
    
    Args:
        attributes (dict): Dizionario degli attributi estratti dal VLM 
                           (es. {"category": "sneakers", "color": "red", "material": "leather"}).
        target_context (str): La scena o il setup fotografico di destinazione.
                              (es. "placed on a wooden table, studio lighting").
                              
    Returns:
        str: Un paragrafo fluido e descrittivo pronto per FLUX.
    """
    # 1. Estrazione sicura della categoria principale (default "object" se mancante)
    category = attributes.get("category", "object").lower()
    
    # 2. Costruzione discorsiva degli attributi principali (Colore e Materiale)
    # T5 preferisce frasi come "A red leather sneakers" piuttosto che "sneakers, red, leather"
    color = attributes.get("color", "")
    material = attributes.get("material", "")
    
    subject_parts = []
    if color:
        subject_parts.append(color)
    if material:
        subject_parts.append(material)
        
    # Uniamo colore e materiale prima della categoria
    subject_modifiers = " ".join(subject_parts)
    main_subject = f"{subject_modifiers} {category}".strip()
    
    # Iniziamo la frase in modo naturale
    prompt_sentences = [f"A high-quality photograph of a {main_subject}."]
    
    # 3. Aggiunta dei dettagli specifici (es. loghi, forme, pattern)
    details = []
    for key, value in attributes.items():
        if key not in ["category", "color", "material"] and value:
            # Sostituiamo gli underscore con spazi per renderlo leggibile a T5
            clean_key = key.replace('_', ' ')
            details.append(f"{clean_key} like {value}")
            
    if details:
        # Es: "It features brand details like logo on the side."
        prompt_sentences.append(f"It features {', and '.join(details)}.")
        
    # 4. Integrazione del target_context in modo fluido
    if target_context:
        # Assicuriamoci che il contesto si leghi bene all'oggetto
        clean_context = target_context.strip()
        if not clean_context.lower().startswith(("placed", "resting", "hanging", "in", "on", "with")):
            clean_context = f"The {category} is {clean_context}"
        else:
            clean_context = f"The {category} is {clean_context}"
            
        prompt_sentences.append(clean_context + ".")
        
    # Uniamo tutto in un singolo paragrafo coeso
    final_prompt = " ".join(prompt_sentences)
    
    # Pulizia di eventuali doppi spazi o punteggiatura ridondante
    final_prompt = " ".join(final_prompt.split())
    final_prompt = final_prompt.replace("..", ".")
    
    return final_prompt

# ==========================================
# ESEMPIO DI UTILIZZO
# ==========================================
if __name__ == "__main__":
    extracted_attributes = {
        "category": "sneakers",
        "color": "red with white stripes",
        "material": "leather",
        "brand_details": "a prominent logo on the side"
    }
    context = "placed on a rustic wooden table, soft shadows, professional studio lighting"
    
    prompt = build_flux_prompt(extracted_attributes, context)
    print(prompt)
    # OUTPUT ATTESO:
    # A high-quality photograph of a red with white stripes leather sneakers. It features brand details like a prominent logo on the side. The sneakers is placed on a rustic wooden table, soft shadows, professional studio lighting.