# [CVPR 2025] HistoFS: Non-IID Histopathologic Whole Slide Image Classification via Federated Style Transfer with RoI-Preserving

## Repository Status
This repository is under active development and updates will be made continously.

### Updates
- **[2025/03/12]** Repository created.
- **[2025/03/12]** Dataset creation process updated.

---

## Dependencies
To ensure smooth execution of the project, install the following dependencies:

```bash
pip install -r requirements.txt
```

Alternatively, install them manually:
```bash
pip install torch torchvision timm pandas numpy scikit-image Pillow openslide-python tqdm argparse
```

### Additional Requirements:
- **Python**: 3.8+
- **CUDA**: (For GPU acceleration, optional)
- **OpenSlide**: Required for whole slide image processing. Install via system package manager:
  - Ubuntu: `sudo apt install openslide-tools`
  - macOS: `brew install openslide`
  - Windows: Download and install from [OpenSlide](https://openslide.org/)

---

## Dataset Preparation
We follow the same patch division and patch feature extraction configuration as previous works.

### Preprocessing Steps
1. Ensure your dataset is structured correctly.
2. Run the dataset preprocessing script:

```bash
python tools/compute_zoomtiler_feats.py
```

This script will:
- Perform patch extraction using DeepZoom tiling.
- Extract patch features for downstream analysis.

---

## Usage
### Training and Evaluation
More details on training and evaluation scripts will be added soon. Stay tuned!

---

## Citation
If you find our work useful, please consider citing:
```bibtex
@inproceedings{YourPaper2025,
  title={HistoFS: Non-IID Histopathologic Whole Slide Image Classification via Federated Style Transfer with RoI-Preserving},
  author={Your Name and Others},
  booktitle={CVPR},
  year={2025}
}
```

---


## License
This repository is released under the **MIT License**. See [LICENSE](LICENSE) for details.
