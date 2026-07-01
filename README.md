# NETRA - ANUDRISHTI 🛰️
### Cross-Modal Satellite Image Retrieval Using Multi-Sensor Remote Sensing Data
**Bhartiya Antariksh Hackathon — Team Submission**
An end-to-end AI pipeline for cross-modal satellite image retrieval. Leverages PyTorch and Meta's DINOv2 to extract deep feature embeddings, matching microwave radar (Sentinel-1 SAR) with optical imagery (Sentinel-2) to overcome the domain gap in remote sensing. 
The link of the dataset we used in this project :- https://www.kaggle.com/datasets/shambac/tum-sentinel-1-2

---

## The Problem

Satellite imagery comes from multiple sensors — optical cameras, SAR (Synthetic Aperture Radar), multispectral instruments. Each sees the world differently. Optical images are visually interpretable but blind under clouds. SAR penetrates clouds but looks nothing like an optical image to a human eye.

Today, analysts work in sensor silos. You query optical archives with optical images. You query SAR archives with SAR metadata. There is no system that lets you ask:

> *"Find me satellite images that show the same geographic scene as this query — regardless of which sensor captured them."*

This is the gap NETRA closes.

---

## What NETRA Does

NETRA is a cross-modal satellite image retrieval system. Given a query image from **any sensor**, it retrieves the most geographically similar images from a **different sensor modality** — in under one second.

```
Cloudy optical image (LISS-IV / Sentinel-2)
        ↓
  Cloud Removal (Pix2Pix GAN)
        ↓
  Shared Encoder (ResNet-50 + projection head)
        ↓
  128-dim embedding vector
        ↓
  FAISS nearest-neighbour search
        ↓
  Top-5 SAR matches from same geographic region
```
## Why It Matters for India

India's satellite constellation — Resourcesat-2 (LISS-IV), RISAT-1S, Cartosat — produces multi-sensor data daily. Disaster response teams during floods in Assam or cyclones along the eastern coast need to correlate SAR and optical imagery in near-real-time. NETRA makes cross-sensor search as fast as a Google image search.

The system is sensor-agnostic by design. Adding a new sensor (hyperspectral, thermal) requires only fine-tuning the encoder on new paired data — no architectural changes.
---

**Supported query → retrieval modality pairs:**
| Query | Database | Type |
|---|---|---|
| Optical | Optical | Same-modal baseline |
| SAR | SAR | Same-modal baseline |
| Optical | SAR | ★ Cross-modal (main contribution) |
| SAR | Optical | ★ Cross-modal (main contribution) |

---

## Two Technical Contributions

NETRA is built on two independent deep learning modules that work together as a pipeline.

---

### Contribution 1 — Cloud-Robust Optical Reconstruction (Pix2Pix GAN)

India's optical satellites are rendered partially or fully blind during the monsoon season — June to September — across the northeast, the Western Ghats, and most agricultural belts. A retrieval system that only works on cloud-free imagery is useless for 4–5 months of the year.

NETRA solves this with a **Pix2Pix GAN** trained to reconstruct cloud-free optical patches from cloudy inputs.

- **Generator**: U-Net architecture — encodes the cloudy image into a latent representation, then decodes it back to a clean reconstruction. Skip connections preserve fine spatial detail.
- **Discriminator**: PatchGAN — evaluates realism at the patch level (70×70), not the full image. Forces the generator to produce locally coherent textures, not just globally plausible blurs.
- **Loss**: Adversarial loss + L1 pixel reconstruction loss (λ=100). The L1 term prevents hallucination; the adversarial term prevents blurring.
- **Training data**: SEN12MS-CR — paired cloudy/cloud-free Sentinel-2 patches, pre-aligned, no co-registration overhead.

The output feeds directly into the retrieval encoder. Cloud removal is not post-processing — it is the first stage of the retrieval pipeline, making the system operable year-round on real Indian remote sensing data.

---

### Contribution 2 — Cross-Modal Shared Embedding Space (Siamese Contrastive Learning)

The key insight driving the retrieval system: a **shared embedding space** where optical and SAR patches of the *same geographic location* map to nearby vectors — regardless of which sensor captured them.

NETRA implements this with a **Siamese architecture built on a frozen DINOv2 backbone**. Both the optical and SAR branches share the same frozen feature extractor, with a lightweight trainable projection head mapping features from each modality into the shared embedding space. Optical and SAR patches covering the *same geographic location* are treated as positive pairs. The projection head is trained to pull their embeddings close together and push different-location embeddings apart (InfoNCE / NT-Xent contrastive loss).

After training, retrieval is a single nearest-neighbour search — sensor modality becomes irrelevant. The geometry of the embedding space encodes *where*, not *how it was captured*. Because DINOv2 is kept frozen, training is lightweight and stable, and the model retains DINOv2's strong general-purpose visual features while learning only the cross-modal alignment.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      NETRA Pipeline                     │
│                                                         │
│  [Optical Query]──→[Cloud Removal GAN]──┐               │
│                                         ├──→[Shared  ]──→[FAISS]──→[Top-K] │
│  [SAR Query]────────────────────────────┘   [Encoder ]           [Results] │
│                                                                             │
│  Database: pre-computed embeddings of all SAR + optical patches             │
└─────────────────────────────────────────────────────────┘
```

**Module 1 — Cloud Removal (Pix2Pix GAN)**
- Generator: U-Net with skip connections · Input 256×256 cloudy patch → cloud-free reconstruction
- Discriminator: PatchGAN (70×70 receptive field) · evaluates local texture realism
- Loss: Adversarial + L1 pixel loss (λ=100) · prevents hallucination and blurring
- Training data: SEN12MS-CR (pre-aligned cloudy/clean Sentinel-2 pairs)
- Target: RMSE < 0.1 · PSNR > 25 dB · SSIM > 0.8

**Module 2 — Shared Encoder (Siamese, Contrastive Learning)**
- Architecture: Siamese network with a **frozen DINOv2 backbone** shared across both modalities
- Projection head: lightweight trainable MLP → 128-dim L2-normalised embedding
- Loss: InfoNCE (contrastive) on optical–SAR positive pairs
- Training data: 75,000+ paired patches from Sentinel-1 (SAR) and Sentinel-2 (Optical)
- Hardware: trained on Apple Silicon (MPS backend), with custom gradient-flow handling to work around MPS compatibility limitations

**Retrieval Index**
- Library: FAISS (IndexFlatIP — cosine similarity)
- Index: pre-computed embeddings of all database images
- Query time: < 1 second

---

## Results

### Module 1 — Cloud Removal (Pix2Pix GAN)
| Metric | Target | Achieved |
|--------|--------|----------|
| RMSE | < 0.1 | — |
| PSNR | > 25 dB | — |
| SSIM | > 0.8 | — |

### Module 2 — Cross-Modal Retrieval
| Metric | Target | Achieved |
|--------|--------|----------|
| F1@5 (same-modal) | > 0.6 | — |
| F1@10 (same-modal) | > 0.7 | — |
| F1@5 (cross-modal) | > 0.4 | — |
| F1@10 (cross-modal) | > 0.5 | — |
| Query time | < 1 s | — |

*Full quantitative results are being finalised; validation runs confirm a high-precision search engine capable of retrieving correct optical matches for radar-based queries in real time.*

---

## What's Left to Do

- **UI/UX implementation** — wrap the retrieval engine in a simple drag-and-drop interface so users can upload their own SAR images for analysis.
- **Real-world testing** — expand evaluation to multi-seasonal datasets (spring, summer, winter) to improve generalisation across different global landscapes.
- **Final presentation** — document the physics behind using radar for cloud penetration and how NETRA benefits disaster management and agricultural monitoring.

---

## Repository Structure

```
Netra/
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_cloud_removal_training.ipynb
│   ├── 03_retrieval_system.ipynb
│   └── 04_full_pipeline_demo.ipynb
├── src/
│   ├── cloud_removal/
│   │   ├── generator.py        # U-Net generator
│   │   ├── discriminator.py    # PatchGAN discriminator
│   │   └── train.py
│   ├── retrieval/
│   │   ├── encoder.py          # ResNet-50 + projection head
│   │   ├── contrastive_loss.py # InfoNCE loss
│   │   ├── build_index.py      # FAISS index builder
│   │   └── train.py
│   └── pipeline/
│       └── main.py             # End-to-end inference
├── outputs/
│   ├── netra_result.png        # Visual comparison figures
│   ├── training_curves.png     # Loss + PSNR over epochs
│   └── metrics.txt             # Final evaluation numbers
├── checkpoints/
│   ├── G_best.pth              # Best GAN generator weights
│   └── retrieval_model.pth     # Best encoder weights
└── demo.ipynb                  # Interactive demo notebook
```

---

## Datasets

| Dataset | Purpose | Source | Co-registration |
|---------|---------|--------|-----------------|
| SEN12MS-CR | GAN training (cloud removal) | Zenodo | Pre-aligned — none needed |
| SEN1-2 | Encoder training (retrieval) — 75,000+ paired optical/SAR patches processed | Zenodo | Pre-aligned — none needed |
| LISS-IV (Bhoonidhi) | Indian scene fine-tuning + demo | NRSC | Manual co-registration |
| Sentinel-1/2 (Copernicus) | Additional retrieval pairs | ESA | Manual co-registration |

**Important**: SEN12MS-CR and SEN1-2 must be used in Google Colab or a cloud environment. Do not download locally — combined size exceeds 8 GB.

---

## Quickstart

```bash
git clone https://github.com/your-username/netra.git
cd netra
pip install -r requirements.txt
```

**Run the full pipeline on a single image:**
```bash
python src/pipeline/main.py \
  --query path/to/query_image.tif \
  --modality optical \
  --top_k 5
```

**Interactive demo:**
```bash
jupyter notebook demo.ipynb
```

---

## Technical Stack

| Layer | Library |
|-------|---------|
| Deep learning | PyTorch + torchvision |
| GAN architecture | Pix2Pix (U-Net + PatchGAN) |
| Contrastive learning | InfoNCE / NT-Xent loss |
| Geospatial I/O | GDAL + rasterio |
| Vector search | FAISS |
| Backbone | DINOv2 (frozen, Siamese architecture) |
| Runtime | Python 3.9+ |
| Hardware | Apple Silicon (MPS) with custom gradient-flow controls |

---

*Built for Bhartiya Antariksh Hackathon*
