"""
pipeline/prompts/dreambench_prompt_compiler.py

Prompt Compiler for Phase 3 (DreamBench zero-shot).


build_dreambench_prompt():
- Prompt 0-19 (recontextualization / accessorization): the scene
  is separated from the noun phrase with ", photographed <scene>". This prevents
  FLUX from interpreting scene attributes (e.g., "blue house", "Eiffel Tower") as 
  properties of the object (attribute bleeding).
- Prompt 20-24 (property modification: "a red {0}", "a cube
    shaped {0}"...): no separation — the modifier must remain attached to the subject
    because it is exactly what the prompt asks to change. These prompts are passed as-is
    with build_dreambench_prompt() using is_property_modification=True.  
"""

import re

# ---------------------------------------------------------------------------
# Mapping key for OBJECTS
# ---------------------------------------------------------------------------

_DETAIL_TEMPLATES_OBJECT = {
    "shape":              "with a {value} shape",
    "pattern":             "featuring a {value} pattern",
    "brand/text":          "with {value} visible",
    "distinct features":  "showing {value}",
}

# ---------------------------------------------------------------------------
# Mapping key for ANIMALS / LIVING
# ---------------------------------------------------------------------------

_DETAIL_TEMPLATES_LIVING = {
    "coat_and_color":       "with {value} coat",
    "facial_features":      "with {value}",
    "distinctive_markings": "showing {value}",
    "accessories":          "wearing {value}",
}

_EXCLUDED_KEYS_OBJECT = {"category", "color", "material", "general", "sdxl_prompt", "_entity_type"}
_EXCLUDED_KEYS_LIVING = {"category", "species_and_breed", "general", "sdxl_prompt", "_entity_type"}

_NEGATIVE_TRIGGERS = [
    "no visible", "none", "n/a", "not readable",
    "unknown", "no brand", "no text",
]

_CONCEPT_TAG_RE = re.compile(r"<[^<>]+>\s*")


def _is_negative_value(value: str) -> bool:
    value_lower = value.strip().lower()
    return any(trigger in value_lower for trigger in _NEGATIVE_TRIGGERS)


def _format_detail(key: str, value: str, templates: dict) -> str:
    template = templates.get(key)
    if template:
        return template.format(value=value)
    return f"with {value}"


def _strip_concept_tag(text: str) -> str:
    """Remove every occurrence of <concept_id> and normalize the spaces."""
    cleaned = _CONCEPT_TAG_RE.sub("", text)
    return " ".join(cleaned.split())


# ---------------------------------------------------------------------------
# Compilators per category (OBJECT / LIVING)
# ---------------------------------------------------------------------------

def _compile_object_phrase(attributes: dict) -> str:
    category = _strip_concept_tag(attributes.get("category", "object")).lower()
    color = attributes.get("color", "")
    material = attributes.get("material", "")

    subject_parts = [p for p in (color, material) if p and not _is_negative_value(p)]
    main_subject = " ".join(subject_parts + [category]).strip()

    details = []
    for key, value in attributes.items():
        if key in _EXCLUDED_KEYS_OBJECT or not value:
            continue
        if isinstance(value, str) and _is_negative_value(value):
            continue
        details.append(_format_detail(key, value, _DETAIL_TEMPLATES_OBJECT))

    phrase = main_subject
    if details:
        phrase = f"{phrase} {', '.join(details)}"

    return _strip_concept_tag(phrase)


def _compile_living_phrase(attributes: dict) -> str:
    species = attributes.get("species_and_breed") or attributes.get("category", "animal")
    species = _strip_concept_tag(species)
    if "," in species:
        parts = [p.strip() for p in species.split(",", 1)]
        if len(parts) == 2:
            species = f"{parts[1]} {parts[0].lower()}"

    details = []
    for key, value in attributes.items():
        if key in _EXCLUDED_KEYS_LIVING or not value:
            continue
        if isinstance(value, str) and _is_negative_value(value):
            continue
        details.append(_format_detail(key, value, _DETAIL_TEMPLATES_LIVING))

    phrase = species
    if details:
        phrase = f"{phrase} {', '.join(details)}"

    return _strip_concept_tag(phrase)


def compile_subject_phrase(fingerprints: dict) -> str:
    """
    Main entry point: given the 'info' dictionary of a concept
    in the database, returns a clean noun phrase in natural language,
    ready to replace {0} in the official DreamBench prompts.

    Output examples:
      OBJECT -> "maroon backpack with colorful patches on the front pocket"
      LIVING -> "Shiba Inu dog with short tan coat, white patch on the left paw"
    """
    entity_type = fingerprints.get("_entity_type", "OBJECT")
    if entity_type == "LIVING":
        phrase = _compile_living_phrase(fingerprints)
    else:
        phrase = _compile_object_phrase(fingerprints)

    phrase = " ".join(phrase.split())
    phrase = phrase.strip(" ,")
    return phrase


def build_dreambench_prompt(template: str, subject_phrase: str, is_property_modification: bool = False) -> str:
    """
    Inject the subject_phrase into the DreamBench template (which uses {0}).

    Prompt 0-19 (recontextualization / accessorization):
        Separate the noun phrase from the scene with ", photographed <scene>".
        This avoids FLUX interpreting environmental attributes (e.g., "blue house",
        "Eiffel Tower") as properties of the object (attribute bleeding).

        Examples:
          "a {0} in the jungle"
            -> "a <phrase>, photographed in the jungle"
          "a {0} with the Eiffel Tower in the background"
            -> "a <phrase>, photographed with the Eiffel Tower in the background"
          "a {0} on top of pink fabric"
            -> "a <phrase>, photographed on top of pink fabric"

    Prompt 20-24 (property modification: "a red {0}", "a cube shaped {0}"...):
        No separation — the modifier must remain attached to the subject
        because it is exactly what the prompt asks to alter.
        Pass is_property_modification=True to use this path.

        Example:
          "a red {0}" -> "a red <phrase>"
    """
    if is_property_modification:
        return template.format(subject_phrase)

    parts = template.split("{0}", 1)
    prefix = parts[0].strip()         
    scene  = parts[1].strip() if len(parts) > 1 else ""

    if not scene:
        return f"{prefix} {subject_phrase}"

    return f"{prefix} {subject_phrase}, photographed {scene}"


# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    object_fp = {
        "general": "<backpack> is a maroon-colored backpack with a front zipper pocket and several patches on its lower front.",
        "category": "backpack",
        "shape": "rectangular with rounded top corners",
        "material": "fabric",
        "color": "maroon",
        "pattern": "none",
        "brand/text": "none",
        "distinct features": "has multiple colorful patches on the front pocket",
        "_entity_type": "OBJECT",
    }

    living_fp = {
        "general": "<dog6> is a Shiba Inu with a thick tan and white coat.",
        "category": "dog",
        "species_and_breed": "Dog, Shiba Inu",
        "coat_and_color": "thick tan and white",
        "facial_features": "dark almond-shaped eyes, triangular upright ears",
        "distinctive_markings": "white patch on the left front paw",
        "accessories": "none",
        "_entity_type": "LIVING",
    }

    obj_phrase = compile_subject_phrase(object_fp)
    live_phrase = compile_subject_phrase(living_fp)

    print("OBJECT phrase:", obj_phrase)
    print("LIVING phrase:", live_phrase)
    print()

    recontex_templates = [
        "a {0} in the jungle",
        "a {0} on top of pink fabric",
        "a {0} with a blue house in the background",
        "a {0} with the Eiffel Tower in the background",
    ]
    print("--- Recontextualization ---")
    for t in recontex_templates:
        print(f"  {build_dreambench_prompt(t, obj_phrase)}")

    print()

    property_templates = [
        "a red {0}",
        "a purple {0}",
        "a cube shaped {0}",
    ]
    print("--- Property modification ---")
    for t in property_templates:
        print(f"  {build_dreambench_prompt(t, obj_phrase, is_property_modification=True)}")