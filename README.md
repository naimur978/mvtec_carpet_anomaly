## Setup

**1. Ensure Python version:**
```
Python 3.14.3
```

**2. Create and activate virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Mac
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Prepare dataset:**
Create a `dataset` folder in the root directory and download `carpet.tar.xz` from [here](https://drive.google.com/file/d/1e0BF8gSs6zflzH2tBUN6vW40UjYv_a8N/view). Extract it into the `dataset` folder.

**5. Verify folder structure:**
After step 4, the project structure should look like:
```
mvtec_carpet_anomaly/
├── dataset/
│   └── carpet.tar.xz
├── notebook.ipynb
├── requirements.txt
├── README.md
├── config.py
└── venv/
```

**6. Run notebook:**
Open `notebook.ipynb` and run all cells. Runtime:
- **CPU:** ~3-4 minutes
- **GPU:** ~2 minutes

## Approach

I wanted to solve **unsupervised anomaly detection and localization** on the MVTec carpet dataset. There are two main ways people approach this:

1. **Feature-Embedding-Based Methods** - Grab features and use distance metrics to spot anomalies
2. **Reconstruction-Based Methods** - Train on normal samples and catch anomalies when reconstruction goes wrong

I decided to build stuff from scratch instead of using pre-made libraries. This way, I'd have full control to experiment and tweak things quickly.

### Methods Explored

**Feature Extraction & Indexing:**
- Cosine and Euclidean distance metrics
- Neighbor size optimization
- Duplicate detection in training data
- Data augmentation techniques

**Feature Extractors:**
- DINOv2
- Qwen feature extractor
- ResNet feature extractor
- Coarse-to-fine feature extraction strategies

**Vector Indexing:**
- Different FAISS indices
- Full vs. small coreset approaches
- KNN-based anomaly detection

**Evaluation Metrics:**
- ROC-AUC (image-level)
- ROC-AUC (pixel-level)
- ROC-Pro
- Otsu thresholding for masking and bounding boxes

**Reconstruction-Based Methods:**
- DDPM (Denoising Diffusion Probabilistic Models)
- RD4AD
- UNet
- DINOv2 autoencoder

### Key Observations

Feature extraction with FAISS indexing and KNN actually works best—beats most other approaches I tried. It consistently delivers solid detection and localization scores without needing heavy computational resources. The simplicity is part of the appeal; it just gets the job done.

The carpet dataset is pretty small and not super diverse. There aren't many 45-degree rotated images or extreme variations, which means there's a natural ceiling on how well any model can generalize. It's a limitation of the data itself, not necessarily the approach. That's something to keep in mind.

I noticed the model's pretty confident on textured defects but struggles with color-based anomalies. Textured surfaces give it way more clues to work with. Pure color shifts are tougher to catch since they don't have that structural information. It makes sense when you think about how the feature extractors work.

Multi-scale features hit better results than single-scale extraction. Makes sense since you catch defects at different levels—some show up at a coarse scale, others need fine details. Mixing both scales gives more robust detections.

With GPU and more time, diffusion models (like Stable Diffusion or Flux) with latent space tweaks or LoRA fine-tuning would probably crush this. But that's a whole different beast that needs serious computational power and paper implementation time. It's on my wishlist for future work.