# [CVPR 2025] HistoFS  
**Non-IID Histopathologic Whole Slide Image Classification via Federated Style Transfer with RoI-Preserving**

---

## 👥 Authors  

Farchan Raswa Hakim, Chun-Shien Lu †, Jia-Ching Wang†
† Corresponding author

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
@inproceedings{Raswa2025HistoFS,
  title     = {HistoFS: Non-IID Histopathologic Whole Slide Image Classification via Federated Style Transfer with RoI-Preserving},
  author    = {Farchan Hakim Raswa and Chun-Shien Lu and Jia-Ching Wang},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2025},
  pages     = {30251--30260}
}

```

---

## 📄 License  
MIT License. See [LICENSE](LICENSE).
