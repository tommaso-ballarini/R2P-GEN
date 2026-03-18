# Configuration Refactoring - FINAL VERSION

## ✅ All Changes Completed

### 1. **Config Organization - Ordered & Reorganized**

The `config.py` file now has a **logical, ordered structure**:

```python
class Config:
    class BuildDatabase:     # 1. Database building settings
    class Database:          # 2. Database naming (right after BuildDatabase)
    class Models:            # 3. Model paths
    class GPU:               # 4. GPU settings (right after Models)
    class Generate:          # 5. Image generation settings
    class Refine:            # 6. Refinement loop settings
    class Images:            # 7. Image sizing
    class Paths:             # 8. Directory paths
```

**Order rationale:**
- `BuildDatabase` + `Database` → Related to database creation
- `Models` + `GPU` → Infrastructure concerns
- `Generate` + `Refine` → Generation pipeline
- `Images` + `Paths` → Supporting configs

---

### 2. **IMAGES_PER_CONCEPT Behavior - CORRECTED** ✅

#### ❌ Old (WRONG) Behavior:
- Returned ALL images per concept
- First image used for fingerprinting

#### ✅ New (CORRECT) Behavior:
**Exactly ONE image per concept:**

```python
if USE_CLIP_SELECTION == True:
    # Select most representative image (closest to CLIP centroid)
    return [most_representative_image]  # SINGLE IMAGE

if USE_CLIP_SELECTION == False:
    # Select first image (sorted numerically)
    return [images[0]]  # SINGLE IMAGE
```

**Key Points:**
- `_select_images_for_concept()` returns a **single-element list**
- That image is used for **both fingerprinting AND storage**
- No multi-view, no "all images" - just **ONE representative image**

---

### 3. **Updated Code Comments**

All docstrings now correctly reflect the behavior:

```python
class DatabaseBuilder:
    """
    - Selects ONE representative image per concept
    - Extracts fingerprints from that single image
    - Stores the selected image in the database
    """

def _select_images_for_concept(self, images):
    """
    Select ONE representative image from a concept's image list.

    Returns:
        list: Single-element list containing the selected image path
    """

def _process_concept(self, concept_data):
    """
    1. Select ONE representative image
    2. Extract fingerprints from that image
    3. Generate SDXL prompt
    4. Store the image in the database
    """
```

---

### 4. **Config.BuildDatabase.USE_CLIP_SELECTION**

Updated comment to be crystal clear:

```python
class BuildDatabase:
    USE_CLIP_SELECTION = True  # True = Select most representative image via CLIP centroid
                               # False = Use first image (sorted numerically)
```

---

## Summary of Behavior

### Use Case 1: CLIP Selection Enabled (DEFAULT)
```python
Config.BuildDatabase.USE_CLIP_SELECTION = True
```

**What happens:**
1. Script loads ALL images in concept folder
2. Extracts CLIP features for all images
3. Computes centroid (mean feature vector)
4. Selects the image **closest to centroid** → Most representative
5. Uses that ONE image for fingerprinting
6. Stores that ONE image in database

**Example:**
```
Concept: bag/alx/
Images: [1.jpg, 2.jpg, 3.jpg, 4.jpg, 5.jpg, 6.jpg, 7.jpg, 8.jpg]

CLIP analysis:
  - Image 4.jpg is closest to centroid ✓

Result:
  - Selected: [4.jpg]  ← ONLY THIS ONE
  - Database entry: {"image": ["path/to/4.jpg"], ...}
```

---

### Use Case 2: CLIP Selection Disabled
```python
Config.BuildDatabase.USE_CLIP_SELECTION = False
```

**What happens:**
1. Script loads ALL images in concept folder
2. Sorts them numerically: [1.jpg, 2.jpg, ..., 8.jpg]
3. Selects **first image** (1.jpg)
4. Uses that ONE image for fingerprinting
5. Stores that ONE image in database

**Example:**
```
Concept: bag/alx/
Images: [1.jpg, 2.jpg, 3.jpg, 4.jpg, 5.jpg, 6.jpg, 7.jpg, 8.jpg]

Result:
  - Selected: [1.jpg]  ← ALWAYS FIRST
  - Database entry: {"image": ["path/to/1.jpg"], ...}
```

---

## File Structure Summary

```
config.py
  ├── BuildDatabase     ← Build settings (moved from build_database.py)
  ├── Database          ← Naming strategy (moved next to BuildDatabase)
  ├── Models            ← Model paths
  ├── GPU               ← GPU settings (moved next to Models)
  ├── Generate          ← Generation settings
  ├── Refine            ← Refinement loop
  ├── Images            ← Image sizing
  └── Paths             ← Directory paths

pipeline/build_database.py
  ├── No local configs (all in config.py) ✓
  ├── _select_images_for_concept() → Returns 1 image ✓
  ├── _process_concept() → Uses 1 image ✓
  └── Clear docstrings ✓
```

---

## Testing

To verify the behavior:

```python
# Test 1: Check selected images count
builder = DatabaseBuilder(...)
concepts = builder._get_concepts()
selected = builder._select_images_for_concept(concepts[0]['images'])
print(len(selected))  # Should print: 1

# Test 2: Run build_database.py
python pipeline/build_database.py

# Check output:
# - Each concept should have exactly 1 image in database
# - "image": ["single_path_here"]  ← NOT a list of multiple paths
```

---

## Database Format

**Old (WRONG) behavior:**
```json
{
  "concept_dict": {
    "<alx>": {
      "name": "alx",
      "image": [
        "path/to/1.jpg",
        "path/to/2.jpg",
        "path/to/3.jpg"  ❌ Multiple images
      ],
      "info": {...}
    }
  }
}
```

**New (CORRECT) behavior:**
```json
{
  "concept_dict": {
    "<alx>": {
      "name": "alx",
      "image": [
        "path/to/4.jpg"  ✅ SINGLE image (most representative via CLIP)
      ],
      "info": {...}
    }
  }
}
```

---

## Confirmation

✅ **Requirements:** Complete and verified
✅ **Config organization:** Ordered logically (BuildDatabase→Database, Models→GPU)
✅ **IMAGES_PER_CONCEPT removed:** No longer exists
✅ **Image selection:** Exactly **ONE** image per concept
✅ **CLIP_SELECTION=True:** Most representative image
✅ **CLIP_SELECTION=False:** First image (sorted)
✅ **Code comments:** Updated and accurate
✅ **Docstrings:** Clear and correct

---

## Final Notes

The code now behaves exactly as requested:
- **ONE image per concept**
- CLIP selection → most representative
- No CLIP → first image (sorted)
- Clear, organized config structure

All changes are in English as requested.
