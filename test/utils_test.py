# test/utils_test.py
"""
Utility functions for test experiments.
"""

import os
import sys
import json
import torch
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from PIL import Image

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def cleanup_gpu():
    """Aggressively free GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def ensure_output_dir(output_dir: str) -> str:
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def load_database(database_path: str, category: str = None, num_concepts: int = None) -> Dict[str, Any]:
    """
    Load database and optionally filter by category.
    
    Args:
        database_path: Path to the database JSON file
        category: Optional category to filter by
        num_concepts: Optional number of concepts to return
        
    Returns:
        Dictionary of concept_id -> content
    """
    if not os.path.exists(database_path):
        raise FileNotFoundError(f"Database not found: {database_path}")
    
    with open(database_path, 'r', encoding='utf-8') as f:
        db = json.load(f)
    
    concept_dict = db.get("concept_dict", {})
    
    if category:
        concept_dict = {
            k: v for k, v in concept_dict.items()
            if v.get("category") == category
        }
    
    if num_concepts:
        concept_ids = list(concept_dict.keys())[:num_concepts]
        concept_dict = {k: concept_dict[k] for k in concept_ids}
    
    return concept_dict


def load_reference_image(image_path: str, size: int = 512) -> Image.Image:
    """
    Load and resize a reference image.
    
    Args:
        image_path: Path to the image
        size: Target size (square)
        
    Returns:
        PIL Image resized to (size, size)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return img


def save_experiment_metadata(
    output_dir: str,
    experiment_name: str,
    config: Dict[str, Any],
    results: Dict[str, Any]
):
    """
    Save experiment metadata to JSON file.
    
    Args:
        output_dir: Output directory
        experiment_name: Name of the experiment
        config: Configuration used
        results: Results dictionary
    """
    metadata = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "results": results
    }
    
    metadata_path = os.path.join(output_dir, "experiment_metadata.json")
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"   💾 Metadata saved to {metadata_path}")


def create_comparison_grid(
    images: List[Image.Image],
    labels: List[str],
    cols: int = 4,
    cell_size: int = 256,
    padding: int = 10,
    label_height: int = 30
) -> Image.Image:
    """
    Create a comparison grid of images with labels.
    
    Args:
        images: List of PIL Images
        labels: List of labels for each image
        cols: Number of columns
        cell_size: Size of each cell
        padding: Padding between cells
        label_height: Height for label text
        
    Returns:
        PIL Image with the grid
    """
    from PIL import ImageDraw, ImageFont
    
    n = len(images)
    rows = (n + cols - 1) // cols
    
    # Calculate grid size
    grid_width = cols * cell_size + (cols + 1) * padding
    grid_height = rows * (cell_size + label_height) + (rows + 1) * padding
    
    # Create grid
    grid = Image.new('RGB', (grid_width, grid_height), color='white')
    draw = ImageDraw.Draw(grid)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    for i, (img, label) in enumerate(zip(images, labels)):
        row = i // cols
        col = i % cols
        
        x = padding + col * (cell_size + padding)
        y = padding + row * (cell_size + label_height + padding)
        
        # Resize image
        img_resized = img.resize((cell_size, cell_size), Image.Resampling.LANCZOS)
        
        # Paste image
        grid.paste(img_resized, (x, y))
        
        # Draw label
        label_y = y + cell_size + 5
        draw.text((x, label_y), label[:30], fill='black', font=font)
    
    return grid


def print_progress_bar(current: int, total: int, prefix: str = "", length: int = 40):
    """Print a simple progress bar."""
    percent = current / total
    filled = int(length * percent)
    bar = '█' * filled + '░' * (length - filled)
    print(f"\r{prefix} |{bar}| {current}/{total} ({percent*100:.1f}%)", end='', flush=True)
    if current == total:
        print()


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        return f"{int(mins)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{int(hours)}h {int(mins)}m"


class ExperimentLogger:
    """Simple logger for experiments."""
    
    def __init__(self, output_dir: str, experiment_name: str):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.log_path = os.path.join(output_dir, "experiment.log")
        self.start_time = datetime.now()
        
        ensure_output_dir(output_dir)
        
        self._log(f"=" * 70)
        self._log(f"EXPERIMENT: {experiment_name}")
        self._log(f"Started: {self.start_time.isoformat()}")
        self._log(f"=" * 70)
    
    def _log(self, message: str):
        """Write to log file and print."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(log_line + "\n")
        
        print(log_line)
    
    def info(self, message: str):
        self._log(f"INFO: {message}")
    
    def success(self, message: str):
        self._log(f"✅ {message}")
    
    def error(self, message: str):
        self._log(f"❌ ERROR: {message}")
    
    def warning(self, message: str):
        self._log(f"⚠️  WARNING: {message}")
    
    def finish(self):
        """Log experiment completion."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        self._log(f"=" * 70)
        self._log(f"EXPERIMENT COMPLETE")
        self._log(f"Duration: {format_time(duration)}")
        self._log(f"=" * 70)


def get_concept_info(content: dict) -> dict:
    """Extract key info from a concept dictionary."""
    info = content.get("info", {})
    return {
        "category": content.get("category", "unknown"),
        "images": content.get("image", []),
        "general": info.get("general", ""),
        "sdxl_prompt": info.get("sdxl_prompt", ""),
        "fingerprints": {
            k: v for k, v in info.items()
            if k not in ["general", "sdxl_prompt"]
        }
    }
