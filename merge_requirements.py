# merge_requirements.py
"""
Merge all requirements files for cluster deployment.

Usage:
    python merge_requirements.py

Output:
    requirements_full.txt - contains all unique dependencies
"""

import re
from pathlib import Path
from datetime import datetime


def parse_requirement(line: str) -> tuple:
    """
    Parse a requirement line into (package_name, full_spec).
    Returns (None, None) for comments and empty lines.
    """
    line = line.strip()
    
    # Skip comments and empty lines
    if not line or line.startswith('#'):
        return None, None
    
    # Handle git URLs (e.g., clip @ git+https://...)
    if '@' in line and 'git+' in line:
        pkg_name = line.split('@')[0].strip()
        return pkg_name.lower(), line
    
    # Handle direct URLs (e.g., flash-attn @ https://...)
    if '@' in line and 'http' in line:
        pkg_name = line.split('@')[0].strip()
        return pkg_name.lower(), line
    
    # Standard package with version specifier
    match = re.match(r'^([a-zA-Z0-9_-]+)', line)
    if match:
        pkg_name = match.group(1).lower()
        return pkg_name, line
    
    return None, None


def merge_requirements():
    """Merge all requirements_*.txt files."""
    
    req_files = [
        "requirements.txt",
        "requirements_baseline.txt",
        "requirements_anydoor.txt",
        "requirements_test.txt"
    ]
    
    # Dictionary to store package -> full requirement line
    packages = {}
    
    # Track which file each package came from
    sources = {}
    
    for req_file in req_files:
        path = Path(req_file)
        if not path.exists():
            print(f"⚠️  Skipping {req_file} (not found)")
            continue
        
        print(f"📖 Reading {req_file}...")
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                pkg_name, full_spec = parse_requirement(line)
                
                if pkg_name and full_spec:
                    # If package already exists, keep the one with version spec
                    if pkg_name in packages:
                        existing = packages[pkg_name]
                        # Prefer lines with version specifiers
                        if '>=' in full_spec or '==' in full_spec:
                            packages[pkg_name] = full_spec
                            sources[pkg_name] = req_file
                    else:
                        packages[pkg_name] = full_spec
                        sources[pkg_name] = req_file
    
    # Write merged requirements
    output_file = "requirements_full.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# requirements_full.txt\n")
        f.write("# ============================================\n")
        f.write("# MERGED REQUIREMENTS FOR CLUSTER DEPLOYMENT\n")
        f.write("# ============================================\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# DO NOT EDIT MANUALLY - regenerate with merge_requirements.py\n")
        f.write("#\n")
        f.write("# Source files:\n")
        for req_file in req_files:
            if Path(req_file).exists():
                f.write(f"#   - {req_file}\n")
        f.write("# ============================================\n\n")
        
        # Sort packages alphabetically
        for pkg_name in sorted(packages.keys()):
            full_spec = packages[pkg_name]
            f.write(f"{full_spec}\n")
    
    print(f"\n✅ Merged {len(packages)} unique dependencies → {output_file}")
    print(f"\n📦 Package sources:")
    
    # Group by source file
    by_source = {}
    for pkg, src in sources.items():
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(pkg)
    
    for src, pkgs in by_source.items():
        print(f"   {src}: {len(pkgs)} packages")
    
    return output_file


if __name__ == "__main__":
    print("\n🔀 Merging Requirements Files\n")
    print("=" * 50)
    merge_requirements()
    print("=" * 50)
    print("\n✅ Done! Use 'pip install -r requirements_full.txt' for cluster deployment.\n")
