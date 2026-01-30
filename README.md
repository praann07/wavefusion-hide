# WaveFusion-Net: HIDE Dataset Training

**Dual-Branch Image Deblurring with Wavelet-Spatial Fusion**

This repository contains the implementation of WaveFusion-Net trained on the **HIDE dataset** for motion deblurring. HIDE (High-quality Image DEblurring) focuses on challenging real-world motion blur scenarios.

---

##  Results

| Metric | Value |
|--------|-------|
| **Best PSNR** | 28.61 dB |
| **Training Epochs** | 30 |
| **Model Parameters** | 9.48M |
| **Training Pairs** | 8,422 |
| **Test Pairs** | 4,050 |

### Validation Performance
- **Epoch 30 Loss**: 0.5545
- **Mean Test PSNR**: 28.35 dB (across 8 samples)

### Loss Components (Final Epoch)
| Component | Value |
|-----------|-------|
| L1 | 0.0253 |
| VGG Perceptual | 1.3141 |
| FFT | 7.6247 |
| Gradient | 0.1589 |
| Wavelet HF | 0.0334 |

---

##  HIDE Dataset Characteristics

The HIDE dataset presents unique challenges:
- **High-resolution images** with complex motion blur
- **Real-world scenarios** (not synthetic)
- **Diverse motion patterns** (camera shake, object motion)
- **Challenging lighting conditions**

This makes HIDE an excellent benchmark for evaluating deblurring generalization.

---

##  Architecture

WaveFusion-Net employs a **dual-branch encoder-decoder** design optimized for HIDE:

### Key Features
1. **Spatial Branch**: NAFNet-style blocks (4→6→6→4 configuration)
2. **Wavelet Branch**: 3-level DWT decomposition with high-frequency attention
3. **Cross-Branch Fusion**: Gated fusion at scales H/2 and H/4
4. **Strip Attention**: 7×1 and 1×7 convolutions for large receptive fields

### Model Specifications
- **Base channels**: 48
- **Total parameters**: 9,484,659 (trainable)
- **Input size**: Variable (padded to multiple of 8)
- **Loss function**: Combined (5 components)

---

## Quick Start

### Prerequisites
```bash
pip install torch torchvision tqdm matplotlib pillow numpy
```

### Dataset Preparation
Download the [HIDE dataset](https://github.com/joanshen0508/HA_deblur) and organize as:
```
/path/to/HIDE_dataset/
├── train.txt (optional: list of blur/gt pairs)
├── test.txt (optional: list of blur/gt pairs)
├── train/
│   ├── blur/ (or recursively discovered)
│   └── GT/
└── test/
    ├── blur/
    └── GT/
```

**Note**: The dataset loader supports:
- Direct `train.txt` / `test.txt` pair lists
- Automatic folder scanning if `.txt` files are missing
- Recursive search for blur/GT images

### Training
1. Open `notebook4151dd896e.ipynb` in Jupyter/Kaggle
2. Update `config['data_root']` to your HIDE path
3. Run all cells

**Training Configuration:**
- Batch size: 4
- Patch size: 256×256
- Epochs: 30 (faster convergence than GoPro)
- Learning rate: 2e-4 → 1e-7 (cosine annealing)
- Optimizer: AdamW (weight decay 1e-4)
- Mixed precision: Enabled
- Validation: Every 10 epochs

---

##  Repository Structure

```
HIDE/
├── notebook4151dd896e.ipynb       # Main training notebook
├── best_model_hide.pth            # Best checkpoint (epoch 30, 28.61 dB)
├── checkpoint_hide_epoch20.pth    # Intermediate checkpoint
├── sample_*_psnrXX.XX.png         # Visual results (8 samples with PSNR)
├── mp2c00029.pdf                  # Research reference
└── README.md                      # This file
```

---

## Training Progress

| Epoch | Loss | PSNR (dB) | Notes |
|-------|------|-----------|-------|
| 1 | - | 27.52 | Initial validation |
| 10 | 0.5837 | 27.52 | Checkpoint saved |
| 20 | 0.5654 | 28.34 | Checkpoint saved |
| 30 | 0.5545 | **28.61** | ✅ Best model saved |

**Training Speed**: ~14 minutes per epoch on 2× GPUs (2105 batches)

---

##  Visual Results

The repository includes 8 comparison images:
- `sample_0_psnr26.62.png`
- `sample_1_psnr25.79.png`
- `sample_2_psnr30.22.png`
- `sample_3_psnr31.14.png`
- `sample_4_psnr27.96.png`
- `sample_5_psnr28.20.png`
- `sample_6_psnr28.64.png`
- `sample_7_psnr28.24.png`

Each image shows: **Blur Input | Deblurred Output | Ground Truth** (side-by-side)

---

## Technical Details

### Robust Dataset Loader
The `HIDEPairs` class features:
- **Flexible path resolution**: Supports absolute/relative paths from `.txt` files
- **Automatic fallback**: Scans folders if `.txt` not found
- **Recursive search**: Discovers images in nested directories
- **GT matching**: Multiple strategies to pair blur/sharp images

### Data Augmentation (Training Only)
- Random 256×256 crops
- Horizontal flips (50% probability)
- Vertical flips (50% probability)

### Loss Function Weights
```python
L1:        1.0
VGG:       0.1
FFT:       0.05
Gradient:  0.1
Wavelet:   0.02
```

---

##  Key Observations

1. **Fast Convergence**: Achieves competitive PSNR in just 30 epochs
2. **Stable Training**: Combined loss prevents mode collapse
3. **Memory Efficient**: 256×256 patches fit batch size 4 on standard GPUs
4. **Robust Validation**: Handles variable-size test images gracefully

---

##  Comparison with Literature

While direct HIDE benchmarks are limited in literature, our 28.61 dB demonstrates:
- Strong performance on challenging real-world blur
- Effective wavelet-spatial fusion
- Good generalization from limited training data

---

## Future Work

- [ ] Train for 50+ epochs to reach ~30 dB threshold
- [ ] Add SSIM tracking during validation
- [ ] Implement loss/metric curve plotting
- [ ] Test on HIDE subset with extreme blur

---

##  Citation

If you use this work, please cite:
```bibtex
@misc{wavefusion2025hide,
  title={WaveFusion-Net: Dual-Branch Image Deblurring on HIDE Dataset},
  author={Your Name},
  year={2025},
  note={30-epoch training achieving 28.61 dB PSNR}
}
```

---



---

## Acknowledgments

- HIDE dataset: [Shen et al.](https://github.com/joanshen0508/HA_deblur)
- NAFNet baseline architecture
- PyTorch community

---

##  Contact

For questions, open an issue or reach out via GitHub.

---

##  Related Work

- [GoPro Training](../GOPRO/) – Same model on synthetic blur
- [RealBlur-J Training](../REALBLUR_J/) – Real camera blur benchmark
