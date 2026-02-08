# 🧪 R2P-GEN Testing Pipeline

This document describes the testing infrastructure for R2P-GEN experiments.

## 📁 Directory Structure

```
R2P-GEN/
├── baseline/                    # Baseline comparison experiments
│   ├── config_baseline.py       # Shared baseline configuration
│   ├── ip_adapter_only.py       # Baseline 1: IP-Adapter only (vanilla)
│   ├── sdxl_prompt_only.py      # Baseline 2: SDXL prompt only (no IP-Adapter)
│   └── general_desc_only.py     # Baseline 3: General description + IP-Adapter
│
├── test/                        # Advanced test experiments
│   ├── config_test.py           # Test configuration + InstantStyle configs
│   ├── utils_test.py            # Shared test utilities
│   ├── setup_anydoor.ps1        # AnyDoor setup script (PowerShell)
│   ├── anydoor_wrapper.py       # Clean wrapper for AnyDoor model
│   ├── anydoor_vanilla.py       # Test 1: AnyDoor without fingerprints
│   ├── anydoor_with_fingerprints.py  # Test 2: AnyDoor + fingerprints
│   └── instantstyle_scaling.py  # Test 3: Layer-wise scaling experiments
│
├── output/                      # Generated outputs (gitignored)
│   ├── baseline/
│   │   ├── ip_adapter_only/{category}/
│   │   ├── sdxl_prompt_only/{category}/
│   │   └── general_desc_only/{category}/
│   └── test/
│       ├── anydoor_vanilla/{category}/
│       ├── anydoor_fingerprints/{category}/
│       └── instantstyle_scaling/
│           ├── {config_name}/{category}/
│           └── comparison/{category}/
│
├── requirements_baseline.txt    # Dependencies for baseline tests
├── requirements_anydoor.txt     # Dependencies for AnyDoor
├── requirements_test.txt        # Dependencies for test utilities
├── requirements_full.txt        # Merged (for cluster deployment)
└── merge_requirements.py        # Script to merge requirements
```

## 🚀 Quick Start

### 1. Setup Branch

```bash
# Create and switch to testing branch
git checkout -b feature/testing-pipeline
```

### 2. Install Dependencies

```bash
# Install test dependencies
pip install -r requirements_test.txt

# For AnyDoor experiments
pip install -r requirements_anydoor.txt

# Or install all at once (for cluster)
python merge_requirements.py
pip install -r requirements_full.txt
```

### 3. Setup AnyDoor (Optional)

```powershell
# Run setup script (Windows PowerShell)
.\test\setup_anydoor.ps1
```

This will:
- Clone AnyDoor repository to `test/external/AnyDoor/`
- Download checkpoint to `checkpoints/anydoor/`

## 📊 Experiments

### Baseline Experiments

Test different generation approaches without the full R2P-GEN pipeline:

```bash
# Baseline 1: IP-Adapter only (vanilla, no optimized prompt)
python baseline/ip_adapter_only.py --category bag --num 5

# Baseline 2: SDXL prompt only (no IP-Adapter)
python baseline/sdxl_prompt_only.py --category bottle --num 10

# Baseline 3: General description + IP-Adapter
python baseline/general_desc_only.py --category cup --num 5
```

**Available categories**: bag, book, bottle, bowl, clothe, cup, decoration, headphone, pillow, plant, plate, remote, retail, telephone, tie, towel, toy, tro_bag, tumbler, umbrella, veg

### AnyDoor Experiments

```bash
# Test 1: AnyDoor without fingerprints
python test/anydoor_vanilla.py --category bag --num 5

# Test 2: AnyDoor with fingerprints
python test/anydoor_with_fingerprints.py --category bag --num 5
```

### InstantStyle Scaling Experiments

```bash
# List available configurations
python test/instantstyle_scaling.py --list-configs

# Compare configurations (prints table)
python test/instantstyle_scaling.py --compare-configs

# Test a single configuration
python test/instantstyle_scaling.py --category bag --num 5 --config v2_high_identity

# Test multiple configurations
python test/instantstyle_scaling.py --category bag --num 5 --config v1_current_baseline v2_high_identity v3_zero_background

# Test ALL configurations
python test/instantstyle_scaling.py --category bag --num 5 --all-configs
```

## 🎨 InstantStyle Configurations

| Config Name | mid | down.b2 max | up.b1 max | Description |
|------------|-----|-------------|-----------|-------------|
| v1_current_baseline | 0.9 | 0.7 | 0.95 | Current R2P-GEN config |
| v2_high_identity | 1.0 | 0.5 | 1.0 | Maximum identity preservation |
| v3_zero_background | 1.0 | 0.0 | 1.0 | Complete background suppression |
| v4_gradual_ramp | 0.6 | 0.4 | 0.97 | Smooth gradient |
| v5_instantstyle_paper | 0.0 | 0.0 | 1.0 | From InstantStyle paper |
| v6_balanced | 0.75 | 0.6 | 0.9 | Balanced approach |
| v7_texture_focus | 0.7 | 0.4 | 0.95 | Focus on texture/material |
| v8_shape_focus | 1.0 | 0.8 | 0.8 | Focus on shape/structure |

## 📈 Analyzing Results

After running experiments, results are organized in `output/`:

1. **Individual images**: `output/{experiment}/{config}/{category}/<concept_id>.png`
2. **Comparison grids**: `output/test/instantstyle_scaling/comparison/{category}/`
3. **Experiment metadata**: `output/{experiment}/{config}/{category}/experiment_metadata.json`
4. **Logs**: `output/{experiment}/{config}/{category}/experiment.log`

## 🔄 Workflow

### Recommended Testing Order

1. **Run baselines first** to establish comparison points:
   ```bash
   python baseline/ip_adapter_only.py --category bag --num 5
   python baseline/sdxl_prompt_only.py --category bag --num 5
   python baseline/general_desc_only.py --category bag --num 5
   ```

2. **Run InstantStyle experiments** to find optimal layer-wise weights:
   ```bash
   python test/instantstyle_scaling.py --category bag --num 5 --all-configs
   ```

3. **Compare results visually** using comparison grids

4. **Run on additional categories** with the best configuration

### Cluster Deployment

```bash
# 1. Merge all requirements
python merge_requirements.py

# 2. Copy requirements_full.txt to cluster

# 3. On cluster:
pip install -r requirements_full.txt

# 4. Run experiments
python baseline/ip_adapter_only.py --category bag --num 10
```

## 📚 References

- **IP-Adapter**: Ye et al., "IP-Adapter: Text Compatible Image Prompt Adapter", arXiv 2023
- **InstantStyle**: Wang et al., "InstantStyle: Free Lunch towards Style-Preserving", arXiv 2024
- **AnyDoor**: Chen et al., "AnyDoor: Zero-shot Object-level Image Customization", CVPR 2024
- **ControlNet**: Zhang et al., "Adding Conditional Control to Text-to-Image Diffusion Models", ICCV 2023

## ⚠️ Notes

- **GPU Memory**: Each experiment requires ~12-16GB VRAM
- **AnyDoor**: Requires separate setup (run `setup_anydoor.ps1`)
- **Checkpoints**: Large model files are gitignored, download separately
- **Output**: All outputs are gitignored, backup important results

## 🐛 Troubleshooting

### "Database not found"
Ensure the database JSON is at `database/database_perva_train_1_clip.json`

### "CUDA out of memory"
- Reduce `--num` to fewer concepts
- Enable more memory optimizations in config

### "AnyDoor not set up"
Run the setup script: `.\test\setup_anydoor.ps1`

### "Module not found"
Install all dependencies: `pip install -r requirements_full.txt`
