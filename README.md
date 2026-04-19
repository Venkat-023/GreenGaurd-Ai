---
title: GreenGuard AI
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# GreenGuard AI

GreenGuard AI is a polished plant-health screening demo focused on potato leaf disease analysis. It combines a U-Net style segmentation model with a downstream disease scoring model to highlight risky regions, estimate severity, and present the result in a clear decision-ready interface.

## Why this version is stronger

- Cleaner product storytelling for hackathons, judges, and recruiters.
- Better Streamlit interface with a branded hero section, severity cards, heatmap view, and downloadable report.
- Safer model loading that supports your local `unet80.h5` instead of forcing a fresh download.
- More honest classifier handling: the CNN is treated as a binary disease-risk model, matching its actual output shape.
- Ready for Hugging Face Spaces with a root Docker deployment setup.

## Live workflow

1. Upload a potato leaf image.
2. Segment suspicious regions with the U-Net model.
3. Score each region for disease risk.
4. Generate a severity summary and exportable text report.

## Project files

- `leaf_disease.py`: upgraded Streamlit app
- `Potato_Disease_Detection_Model2.h5`: disease scoring model
- `unet80.h5`: segmentation model
- `.streamlit/config.toml`: Streamlit runtime configuration
- `Dockerfile`: Hugging Face Spaces container entrypoint

## Model placement

The app looks for the U-Net model in this order:

1. `GREEN_GUARD_UNET_PATH`
2. `./unet80.h5`
3. `C:/Users/admin/Downloads/unet80.h5`

For deployment on Hugging Face, place `unet80.h5` in the repository root so the Space can load it directly.

## Local run

```bash
pip install -r requirements.txt
streamlit run leaf_disease.py
```

## Hugging Face deployment

This repository is prepared for a Docker Space. Once you share the Hugging Face repo link, the remaining deployment flow is:

1. Copy or commit `unet80.h5` into the repo root.
2. Push the repository to your Hugging Face Space.
3. Let Hugging Face build the Docker app on port `7860`.

## Notes

- The classifier model currently behaves like a binary disease-risk predictor.
- The UI and messaging are framed to present a strong, coherent demo without overstating unsupported class granularity.
