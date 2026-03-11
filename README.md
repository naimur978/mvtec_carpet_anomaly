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

I tackled **unsupervised anomaly detection and localization** on the MVTec carpet dataset. Two principal paradigms exist in this domain:

1. **Feature-Embedding-Based Methods** - Extract features and use distance-based approaches
2. **Reconstruction-Based Methods** - Learn to reconstruct normal samples and detect anomalies via reconstruction error

Rather than relying on existing libraries, I built custom solutions to maximize control and flexibility for rapid experimentation.

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

- **Best Performance:** Feature extraction with vector indexing (FAISS) and KNN provides superior evaluation scores across both detection and localization tasks
- **Dataset Limitations:** The carpet dataset is small and lacks diversity (e.g., few 45-degree rotated images), which may limit generalization
- **Texture Sensitivity:** I observed the model shows higher confidence on textured defects; color-based anomalies are harder to detect
- **Coarse-to-Fine:** I found that extracting features at multiple scales improves detection performance
- **Future Potential:** I believe diffusion models (e.g., Stable Diffusion, Flux) with latent space or LoRA fine-tuning could yield better results with sufficient computational resources