"""
pipeline/prompts/dreambench_prompts.py

Official DreamBooth prompts for the DreamBench benchmark (25 prompts per subject type, OBJECTS and LIVING).

"""

# ---------------------------------------------------------------------------
# OBJECT — 21 out of 30 subjects DreamBench
# ---------------------------------------------------------------------------

OBJECT_PROMPTS = [
    # --- Recontextualization (20) ---
    "a {0} in the jungle",
    "a {0} in the snow",
    "a {0} on the beach",
    "a {0} on a cobblestone street",
    "a {0} on top of pink fabric",
    "a {0} on top of a wooden floor",
    "a {0} with a city in the background",
    "a {0} with a mountain in the background",
    "a {0} with a blue house in the background",
    "a {0} on top of a purple rug in a forest",
    "a {0} with a wheat field in the background",
    "a {0} with a tree and autumn leaves in the background",
    "a {0} with the Eiffel Tower in the background",
    "a {0} floating on top of water",
    "a {0} floating in an ocean of milk",
    "a {0} on top of green grass with sunflowers around it",
    "a {0} on top of a mirror",
    "a {0} on top of the sidewalk in a crowded street",
    "a {0} on top of a dirt road",
    "a {0} on top of a white rug",
    # --- Property / color modification (5) ---
    "a red {0}",
    "a purple {0}",
    "a shiny {0}",
    "a wet {0}",
    "a cube shaped {0}",
]

# ---------------------------------------------------------------------------
# LIVING — 9 out of 30 subjects DreamBench (dogs/cats)
# ---------------------------------------------------------------------------

LIVING_PROMPTS = [
    # --- Recontextualization (10) ---
    "a {0} in the jungle",
    "a {0} in the snow",
    "a {0} on the beach",
    "a {0} on a cobblestone street",
    "a {0} on top of pink fabric",
    "a {0} on top of a wooden floor",
    "a {0} with a city in the background",
    "a {0} with a mountain in the background",
    "a {0} with a blue house in the background",
    "a {0} on top of a purple rug in a forest",
    # --- Accessorization (10) ---
    "a {0} wearing a red hat",
    "a {0} wearing a santa hat",
    "a {0} wearing a rainbow scarf",
    "a {0} wearing a black top hat and a monocle",
    "a {0} in a chef outfit",
    "a {0} in a firefighter outfit",
    "a {0} in a police outfit",
    "a {0} wearing pink glasses",
    "a {0} wearing a yellow shirt",
    "a {0} in a purple wizard outfit",
    # --- Property modification (5) ---
    "a red {0}",
    "a purple {0}",
    "a shiny {0}",
    "a wet {0}",
    "a cube shaped {0}",
]



PROPERTY_MODIFICATION_INDICES = set(range(20, 25))


def is_property_modification_prompt(prompt_idx: int) -> bool:
    return prompt_idx in PROPERTY_MODIFICATION_INDICES


def get_prompts_for_entity_type(entity_type: str) -> list[str]:
    """
    Returns the list of 25 prompts corrected based on the entity type classified during fingerprint extraction 
    ('_entity_type' in the JSON).
    """
    if entity_type == "LIVING":
        return LIVING_PROMPTS
    return OBJECT_PROMPTS


if __name__ == "__main__":
    print(f"OBJECT_PROMPTS: {len(OBJECT_PROMPTS)} prompt")
    print(f"LIVING_PROMPTS: {len(LIVING_PROMPTS)} prompt")
    assert len(OBJECT_PROMPTS) == 25
    assert len(LIVING_PROMPTS) == 25
    print("OK: 25/25 for both categories.")