"""
Packaging script for Google Colab offloading.
Creates:
1. sasol_code_bundle.zip (~50KB) - contains all source code, configs, label rules, test.csv, SampleSubmission.csv, notebook
2. sasol_full_bundle.zip (~25MB) - includes data/raw/train.csv for one-click upload to Colab/Google Drive
"""

import os
import zipfile
from pathlib import Path

def create_bundle(include_train_csv: bool = True):
    root = Path(".")
    zip_name = "sasol_full_bundle.zip" if include_train_csv else "sasol_code_bundle.zip"
    
    files_to_pack = [
        "requirements.txt",
        "train.py",
        "run_eda.py",
        "configs/base_config.yaml",
        "data/raw/label_rules.py",
        "data/raw/label_config.json",
        "data/raw/DATA_DICTIONARY.md",
        "data/raw/LABEL_RULES.md",
        "data/raw/SampleSubmission.csv",
        "data/raw/test.csv",
    ]
    
    if include_train_csv:
        files_to_pack.append("data/raw/train.csv")
        
    # Add all files in src/
    for p in (root / "src").rglob("*.py"):
        files_to_pack.append(str(p.relative_to(root)))
        
    # Add notebooks
    for p in (root / "notebooks").rglob("*.ipynb"):
        files_to_pack.append(str(p.relative_to(root)))
        
    print(f"Creating {zip_name} with {len(files_to_pack)} items...")
    with zipfile.ZipFile(zip_name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files_to_pack:
            fp = Path(f)
            if fp.exists():
                print(f"  + Adding {f}")
                zf.write(fp, arcname=f)
            else:
                print(f"  ! Warning: {f} not found, skipping.")
                
    size_mb = Path(zip_name).stat().st_size / (1024 * 1024)
    print(f"Done! {zip_name} created successfully ({size_mb:.2f} MB)")

if __name__ == "__main__":
    import sys
    inc_train = "--no-train" not in sys.argv
    create_bundle(include_train_csv=inc_train)
