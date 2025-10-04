# Extracted Modules

This document tracks code modules that were extracted from this repository to maintain scope alignment.

## Histopathology CNN/ViT/MIL Module

**Extracted on**: 2025-10-03  
**Extract bundle**: `../histopath-extract.bundle`  
**Target repository**: `altalanta/histopath-mets-cnn-vit-mil`  
**Reason**: Computer vision models for medical imaging - completely out of scope for nutrition analysis pipeline

### Original paths:
- `histopath-mets-cnn-vit-mil/src/histopath/training/models.py`
- `histopath-mets-cnn-vit-mil/src/histopath/training/__init__.py`

### To create the new repository:
```bash
git clone ../histopath-extract.bundle histopath-mets-cnn-vit-mil
cd histopath-mets-cnn-vit-mil
git remote add origin git@github.com:altalanta/histopath-mets-cnn-vit-mil.git
git push -u origin main
```

This extraction preserves the complete git history for the histopathology module.