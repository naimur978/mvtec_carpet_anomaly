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

## My Approach

This is an **unsupervised anomaly detection problem**. There is essentially one normal class to model. The dataset contains **89 defective images (1K resolution)**. With such a small number of anomalous samples, there is a clear risk of overfitting if a supervised approach is used. However, in real industrial environments anomalies are typically rare, so having 89 defective samples in a controlled dataset is not unreasonable.

My plan was to first determine **whether an image is anomalous or not**, and then **generate a heatmap to localize the anomaly**. Ground truth masks are provided for the test set, which allows evaluation at the end of the pipeline. However, these masks cannot be used during training. Therefore, for localization I needed to design a **thresholding strategy to generate masks from anomaly scores**, which I will explain later in this document.

There are two primary ways to approach anomaly detection in this context:

1. **Feature-Embedding-Based Methods**
2. **Reconstruction-Based Methods**

I chose to build most components **from scratch** instead of relying on pre-built frameworks such as *anomalib*. This gave me full control over the pipeline and allowed me to experiment and iterate quickly. My initial goal was to achieve reasonable performance using traditional methods. If that was insufficient, I planned to move toward approaches inspired by more recent research papers.

---

## Architecture

![Architecture Diagram](assets/architecture.png)

The pipeline starts with the **normal ("good") training images**. I experimented with several augmentation strategies. However, since the images were captured from a very consistent angle and under controlled conditions, augmentations did not improve performance and in some cases even degraded it.

This observation is consistent with findings reported in [Revisiting Reverse Distillation for Anomaly Detection](https://arxiv.org/pdf/2304.03294).

The authors note that *combining multiple augmentation methods does not necessarily improve anomaly detection accuracy*.

I experimented with different **feature extraction approaches**, including both transformer-based and traditional CNN-based methods. Ultimately, **DINOv2** provided the most stable and effective features. I also tested **DINOv3**, but it did not improve performance and required significantly more computation time.

Additionally, I experimented with **multi-scale feature extraction** instead of using only fixed **224×224 inputs**, which improved the ROC-AUC score.

For **vector indexing**, I evaluated several FAISS-based indices such as **HNSW** and **IVF**. In my experiments, **L2-based indexing performed faster and more reliably**. I also briefly evaluated **ScaNN**.

Anomaly scoring is performed using **k-nearest neighbors (kNN)** in the feature embedding space.

Evaluation is conducted using:

- **ROC-AUC (image-level)**
- **ROC-AUC (pixel-level)**
- **ROC-PRO**



## Other Methods I Explored

### Feature Extraction & Indexing

- Cosine and Euclidean distance metrics
- Neighbor size optimization
- Duplicate detection in training data
- Data augmentation techniques

### Feature Extractors

- DINOv2
- Qwen feature extractor
- ResNet feature extractor
- Coarse-to-fine feature extraction strategies

### Vector Indexing

- Different FAISS indices
- Full memory bank vs. coreset sampling
- kNN-based anomaly detection

### Evaluation Metrics

- ROC-AUC (image-level)
- ROC-AUC (pixel-level)
- ROC-PRO
- Otsu thresholding for mask generation and bounding boxes

### Reconstruction-Based Methods

- DDPM (Denoising Diffusion Probabilistic Models)
- RD4AD
- U-Net
- DINOv2 autoencoder





### Results



| Method | Pixel ROC-AUC | AU-PRO | Image AUROC | FPS |
|--------|---------------|--------|-------------|-----|
| ResNet Feature Extraction | 0.9339 | 0.6720 | 0.9446 | 3.35 |
| DDPM Diffusion | – | – | 0.5313 | – |
| RD4AD | – | – | 0.5173 | – |
| U-Net Reconstruction | – | – | 0.5269 | – |
| DINOv2 AutoEncoder | – | – | 0.8431 | – |
| Cosine + Flip Augmentation | 0.9914 | 0.9339 | 1.0000 | 6.82 |
| **Coarse-to-Fine + FAISS + kNN** | **0.9915** | **0.9337** | **1.0000** | **~6.5** |

I put some of my draft notebooks on these models under "assets" folder.


For anomaly localization, I ultimately used **Otsu’s Thresholding method** to convert the anomaly heatmap into binary masks. After experimenting with several thresholding strategies, this method proved to be the most **reliable and consistent with the ROC-AUC results**.

![Mask](assets/mask.png)

Normal images consistently produce lower anomaly scores than defective ones, which allows me to set an optimized threshold based on roc curve that reliably distinguishes them. 

![Threshold](assets/threshold.png)

Obviously the features are more affected by the textures, which is why i am getting better anomaly scores for "cut" and "hole". Which is why, "color" defects gave worse score compared to its peers, cz of lack of  geometric or texture disruptions.

![Defects](assets/defects.png)

Right now, in my code, i am choosing patches randomly, i wanted to see how sparse they are. They look good enough, but the one i am trying is random, i could retain most representative and diverse patches using algorithms like greedy coreset. But roc-auc score is already good enough right now, so didnt' change much here.

![Patches](assets/patches.png)

### Observations

Feature extraction with FAISS indexing and KNN actually works best—beats most other approaches I tried. It consistently delivers solid detection and localization scores without needing heavy computational resources. The simplicity is part of the appeal; it just gets the job done.

The carpet dataset is pretty small and not super diverse. There aren't many 45-degree rotated images or extreme variations, which means there's a natural ceiling on how well any model can generalize. It's a limitation of the data itself, not necessarily the approach. That's something to keep in mind.

I noticed the model's pretty confident on textured defects but struggles with color-based anomalies. Textured surfaces give it way more clues to work with. Pure color shifts are tougher to catch since they don't have that structural information. It makes sense when you think about how the feature extractors work.

Multi-scale features hit better results than single-scale extraction. Makes sense since you catch defects at different levels—some show up at a coarse scale, others need fine details. Mixing both scales gives more robust detections.



TTA (Test Time Augmentation) could make the inference robust, but when I tried it didnt improve that much, because as I said earlier augmentation works better when the nature of the picture is uncertain, but the dataset i have seems taken in controlled environment with proper cropping. TTA would work better if the images would be taken from random angles. Here, geometrice augmentation introduced noise rather than providing useful variation.


## Limitations



## What I would do

With GPU and more time, diffusion models (like Stable Diffusion or Flux) with latent space tweaks or LoRA fine-tuning would probably crush this. But that's a whole different beast that needs serious computational power and paper implementation time. It's on my wishlist for future work.

Here is the presentation [presentation](assets/mvtec.pptx) file.
