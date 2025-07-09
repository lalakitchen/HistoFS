# [CVPR 2025] HistoFS  
**Non-IID Histopathologic Whole Slide Image Classification via Federated Style Transfer with RoI-Preserving**

---

## 🔧 Repository Status  
Active development. Recent updates:
- **2025/07/09**: Added training scripts (`experiments/`)
- **2025/07/09**: Added test-time evaluation (`evaluations/evaluation.py`)
- **2025/07/09**: Models saved to `model_checkpoints/`

---

## 📦 Installation

```bash
pip install torch torchvision timm pandas numpy scikit-image Pillow openslide-python tqdm argparse
```

Additional:
- Python ≥ 3.8  
- OpenSlide:  
  - Ubuntu: `sudo apt install openslide-tools`  
  - macOS: `brew install openslide`  
  - Windows: [openslide.org](https://openslide.org/)

---

## 📁 Dataset & Preprocessing

```bash
python tools/compute_zoomtiler_feats.py
```

- Divides WSI into patches
- Extracts features for training

---

## 🎯 Style Generation

```bash
python tools/pseudo_bag_style_generation.py --FEATS_TYPE ssl_vit --dataset c17 --NUM_PSEUDO_STYLE 5
```

- Applies Wasserstein K-means to extract style centroids

---

## 🚀 Training

```bash
# HER2
python experiments_her2/train_her2_our.py --dataset her2

# TCGA RCC
python experiments_rcc/train_rcc_our.py --dataset tcga_rcc

# C17
python experiments_c17/train_c17_our.py --dataset c17
```

Trained models will be saved to `model_checkpoints/`.

---

## ✅ Evaluation

```bash
python evaluations/evaluation.py --dataset tcga_rcc --backbone dino --federated FedAvg --style Our
```

---

## 📖 Citation

```bibtex
@inproceedings{YourPaper2025,
  title={HistoFS: Non-IID Histopathologic Whole Slide Image Classification via Federated Style Transfer with RoI-Preserving},
  author={Your Name and Others},
  booktitle={CVPR},
  year={2025}
}
```

---

## 📄 License  
MIT License. See [LICENSE](LICENSE).
