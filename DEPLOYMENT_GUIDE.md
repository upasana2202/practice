# Deployment Guide: Components & Licensing

## 📦 Required Components from pytorch-CycleGAN-and-pix2pix

Based on your web application (`web_app.py`), here are the **specific components** you need to ship:

### Core Files (REQUIRED)

#### 1. Models Directory

```
models/
├── __init__.py              # Required for module imports
├── base_model.py            # Base class for all models
├── pix2pix_model.py         # Pix2pix model implementation (YOUR USE CASE)
└── networks.py              # Network architectures (U-Net, etc.)
```

**NOT NEEDED:**

- ❌ `cycle_gan_model.py` - You're using pix2pix only
- ❌ `colorization_model.py` - Not used in your app
- ❌ `test_model.py` - Not used in your app
- ❌ `template_model.py` - Template only

#### 2. Options Directory

```
options/
├── __init__.py              # Required for module imports
├── base_options.py          # Base options class
└── test_options.py          # Test-time options (used by web_app.py)
```

**NOT NEEDED:**

- ❌ `train_options.py` - Only needed for training

#### 3. Util Directory

```
util/
├── __init__.py              # Required for module imports
└── util.py                  # Utility functions (used by base_options.py)
```

**NOT NEEDED:**

- ❌ `html.py` - HTML visualization (not used in your app)
- ❌ `image_pool.py` - Training only
- ❌ `visualizer.py` - Training visualization
- ❌ `get_data.py` - Dataset utilities

#### 4. Your Custom Files

```
web_app.py                   # Your Flask application
templates/
└── index.html               # Your web interface
requirements_webapp.txt       # Dependencies
start_webapp.bat             # Startup script (Windows)
```

#### 5. Your Trained Models

```
pretrained_models/
├── G_100e_cleanT_L65_Lr25.pth   # Your custom trained weights
├── G_250e_cleanT_L50_Lr10.pth   # (These are NOT from the original repo)
├── G_250e_cleanT_L85_Lr22.pth
└── G_300e_dirtyT_L100_Lr20.pth
```

### Data Directory (OPTIONAL)

You may need minimal parts if your code references them:

```
data/
└── __init__.py              # Might be imported by options/models
```

---

## 📋 Minimal Distribution Package Structure

```
your-sketch-to-photo-app/
├── LICENSE                          # YOUR LICENSE (can be anything)
├── THIRD_PARTY_LICENSES.txt         # Attribution to original authors
├── README.md                        # Your documentation
├── requirements_webapp.txt          # Dependencies
├── web_app.py                       # Your wrapper
├── start_webapp.bat                 # Startup script
│
├── models/                          # From original repo
│   ├── __init__.py
│   ├── base_model.py
│   ├── pix2pix_model.py
│   └── networks.py
│
├── options/                         # From original repo
│   ├── __init__.py
│   ├── base_options.py
│   └── test_options.py
│
├── util/                            # From original repo
│   ├── __init__.py
│   └── util.py
│
├── data/                            # From original repo (minimal)
│   └── __init__.py
│
├── templates/                       # Your frontend
│   └── index.html
│
└── pretrained_models/               # Your trained weights
    ├── G_100e_cleanT_L65_Lr25.pth
    ├── G_250e_cleanT_L50_Lr10.pth
    ├── G_250e_cleanT_L85_Lr22.pth
    └── G_300e_dirtyT_L100_Lr20.pth
```

**Total size:** ~7 files from original repo + your additions

---

## 🔍 Dependency Chain Analysis

```
web_app.py
├─→ models.pix2pix_model (Pix2PixModel)
│   ├─→ models.base_model (BaseModel)
│   │   └─→ models.networks
│   └─→ models.networks
│
├─→ options.test_options (TestOptions)
│   └─→ options.base_options (BaseOptions)
│       ├─→ util.util
│       ├─→ models (module check)
│       └─→ data (module check)
│
└─→ torchvision.transforms (PyTorch - separate dependency)
```

---

## ⚖️ Licensing & Distribution

### What's Covered by BSD License

✅ **From original repo** (must keep BSD license notice):

- `models/` directory files
- `options/` directory files
- `util/` directory files
- `data/__init__.py`

### What's YOURS (can license however you want)

✅ **Your original work**:

- `web_app.py` - Your Flask wrapper
- `templates/index.html` - Your frontend
- `WEB_APP_README.md` - Your documentation
- `ARCHITECTURE.md` - Your documentation
- `start_webapp.bat` - Your script
- `requirements_webapp.txt` - Your dependency list
- All `.pth` model weight files - **YOUR TRAINED MODELS**

---

## 📝 Recommended LICENSE File

```markdown
# Your Project License

Copyright (c) 2025, [Your Name]

[YOUR CHOSEN LICENSE - MIT, Apache 2.0, Proprietary, etc.]

---

## Third-Party Components

This project incorporates code from pytorch-CycleGAN-and-pix2pix, which is
licensed under the BSD 2-Clause License. See THIRD_PARTY_LICENSES.txt for
complete license text.

The following directories contain third-party code:

- models/
- options/
- util/
- data/**init**.py

Copyright (c) 2017, Jun-Yan Zhu and Taesung Park
```

---

## 🚀 Distribution Options

### Option 1: GitHub Repository (Recommended)

```
your-sketch2photo/
├── LICENSE                    # Your license
├── THIRD_PARTY_LICENSES.txt   # BSD attribution
├── README.md
├── [all required files above]
└── .gitignore                 # Exclude large .pth files or use Git LFS
```

### Option 2: PyPI Package

```python
# setup.py
setup(
    name='your-sketch2photo',
    license='MIT',  # Your choice
    packages=['models', 'options', 'util', 'data'],
    install_requires=['flask', 'torch', 'torchvision', ...],
)
```

### Option 3: Docker Container

```dockerfile
FROM python:3.11
COPY models/ /app/models/
COPY options/ /app/options/
COPY util/ /app/util/
COPY web_app.py /app/
COPY templates/ /app/templates/
COPY THIRD_PARTY_LICENSES.txt /app/
# ... rest of your Dockerfile
```

### Option 4: Standalone Executable (PyInstaller)

Package everything into a single .exe with proper license files included.

---

## ✅ Compliance Checklist

Before distributing, ensure:

- [ ] `THIRD_PARTY_LICENSES.txt` includes full BSD license text
- [ ] Copyright notices retained in all original source files
- [ ] Your `README.md` credits the original authors
- [ ] Clear separation between your code and their code (if possible)
- [ ] Your trained model weights are clearly identified as yours
- [ ] Dependencies listed in `requirements.txt`
- [ ] Your LICENSE file clarifies which parts are yours

---

## 🎯 Files You DON'T Need to Ship

**Training Related:**

- ❌ `train.py`
- ❌ `test.py`
- ❌ `datasets/` (dataset preparation scripts)
- ❌ `scripts/train_*.sh`

**Documentation (optional):**

- ❌ `docs/` directory
- ❌ `README.md` (from original repo)
- ❌ `.ipynb` notebooks

**Other Models:**

- ❌ `models/cycle_gan_model.py`
- ❌ `models/colorization_model.py`
- ❌ `models/test_model.py`
- ❌ `models/template_model.py`

**Testing:**

- ❌ `test_setup.py`
- ❌ `scripts/test_*.sh`

**Dataset Processing:**

- ❌ All files in `data/` except `__init__.py`
- ❌ `datasets/` entire directory

---

## 📊 Size Comparison

**Full Original Repo:** ~150+ files  
**Your Minimal Distribution:** ~15-20 files + your additions  
**Space Saved:** ~85-90% smaller

---

## 🔗 Attribution Examples

### In Your README.md:

```markdown
## Acknowledgments

This project uses the pix2pix model architecture from:

- **pytorch-CycleGAN-and-pix2pix** by Jun-Yan Zhu and Taesung Park
- Source: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
- Licensed under BSD 2-Clause License (see THIRD_PARTY_LICENSES.txt)

## Citation

If you use this work, please cite both this project and the original pix2pix paper:

\`\`\`bibtex
@inproceedings{pix2pix2017,
title={Image-to-Image Translation with Conditional Adversarial Networks},
author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
booktitle={CVPR},
year={2017}
}
\`\`\`
```

### In Your Web App UI:

Add a footer or about page:

```html
<footer>
  Powered by pix2pix (Zhu & Park, 2017) |
  <a href="https://github.com/your/repo">Source Code</a>
</footer>
```

---

## ⚠️ Important Notes

1. **Model Weights Are Yours**: Your trained `.pth` files are NOT covered by their license
2. **Code Architecture IS Theirs**: The model/options/util code structure is BSD licensed
3. **Your Wrapper Is Yours**: `web_app.py` and templates are your original work
4. **Commercial Use OK**: BSD allows commercial use with attribution
5. **No Warranty**: Both BSD and most open licenses disclaim warranties

---

## 🎉 Summary

**What You Must Do:**

1. Include `THIRD_PARTY_LICENSES.txt` with BSD license text
2. Keep copyright notices in their source files
3. Credit them in your README

**What You Can Do:**

1. License your wrapper code under ANY license (MIT, GPL, proprietary, etc.)
2. Sell your application commercially
3. Modify their code
4. Keep your model weights proprietary if desired

**What You Cannot Do:**

1. Remove their copyright notices from their code
2. Claim you wrote the pix2pix architecture
3. Use their name to endorse your product without permission

---

_Generated: October 2025_
