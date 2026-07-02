"""
Construction of the FLUX prompt (Textual Anchoring), starting from the attributes
extracted from the VLM.
"""

_DETAIL_TEMPLATES = {
    "shape":             "with a {value} shape",
    "pattern":           "featuring a {value} pattern",
    "brand/text":        "with {value} visible",
    "distinct features": "showing {value}",
    "soft tags":         "also known as {value}",
    "state":             "{value}",
}

_EXCLUDED_KEYS = {"category", "color", "material", "general", "sdxl_prompt"}


_NEGATIVE_TRIGGERS = [
    "no visible", "none", "n/a", "not readable",
    "unknown", "no brand", "no text",
]


def _is_negative_value(value: str) -> bool:
    """True if the value indicates a missing/undetected field in the VLM."""
    value_lower = value.strip().lower()
    return any(trigger in value_lower for trigger in _NEGATIVE_TRIGGERS)


def _format_detail(key: str, value: str) -> str:
    """
    Transforms a (key, value) pair from the fingerprint into a natural language sentence.

    Uses _DETAIL_TEMPLATES if the key is known, otherwise applies a
    generic fallback that keeps the value but discards the raw key name.
    """
    template = _DETAIL_TEMPLATES.get(key)
    if template:
        return template.format(value=value)
    return f"with {value}"


def build_flux_prompt(attributes: dict, target_context: str) -> str:
    """
    Builds a prompt in natural language optimized for the T5 encoder of FLUX,
    by combining a dictionary of physical attributes (Textual Anchoring) with a target context.

    Args:
        attributes (dict): Dictionary of attributes extracted from the VLM
                           (e.g., {"category": "sneakers", "color": "red", "material": "leather"}).
        target_context (str): The photographic scene or setup of destination.
                              (e.g., "placed on a wooden table, studio lighting").

    Returns:
        str: A fluent and descriptive paragraph ready for FLUX.
    """
    # 1. Safe extraction of the main category (default "object" if missing)
    category = attributes.get("category", "object").lower()

    # 2. Construction of the main subject description (Color and Material)
    color = attributes.get("color", "")
    material = attributes.get("material", "")

    subject_parts = []
    if color:
        subject_parts.append(color)
    if material:
        subject_parts.append(material)

    subject_modifiers = " ".join(subject_parts)
    main_subject = f"{subject_modifiers} {category}".strip()

    prompt_sentences = [f"A high-quality photograph of a {main_subject}."]

    # 3. Adding specific details (e.g., logos, shapes, patterns)
    details = []
    for key, value in attributes.items():
        if key in _EXCLUDED_KEYS or not value:
            continue
        if isinstance(value, str) and _is_negative_value(value):
            continue
        details.append(_format_detail(key, value))

    if details:
        prompt_sentences.append(f"It features {', and '.join(details)}.")

    if target_context:
        clean_context = target_context.strip()
        if not clean_context.lower().startswith(("placed", "resting", "hanging", "in", "on", "with")):
            clean_context = f"The {category} is {clean_context}"
        else:
            clean_context = f"The {category} is {clean_context}"

        prompt_sentences.append(clean_context + ".")

    final_prompt = " ".join(prompt_sentences)

    final_prompt = " ".join(final_prompt.split())
    final_prompt = final_prompt.replace("..", ".")

    return final_prompt


# ==========================================
# Usage example (for testing purposes)
# ==========================================
if __name__ == "__main__":
    extracted_attributes = {
        "category": "sneakers",
        "color": "red with white stripes",
        "material": "leather",
        "shape": "low-top with rounded toe",
        "pattern": "no visible pattern",
        "brand/text": "a prominent Nike logo on the side",
        "distinct features": "a small scratch near the heel",
        "general": "<abc> is a red leather sneaker with white stripes.",
        "sdxl_prompt": "(red leather sneakers:1.3), Nike logo, ...",
    }
    context = "placed on a rustic wooden table, soft shadows, professional studio lighting"

    prompt = build_flux_prompt(extracted_attributes, context)
    print(prompt)
