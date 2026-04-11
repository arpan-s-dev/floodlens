# Space Disaster Mapper - Research Summary

This project implements and extends existing space/earth-observation ML research. We are not claiming to invent Prithvi, Sen1Floods11, ImpactMesh, SatMAE, or AstroNet. The original value of this repo is a reproducible pipeline, baseline comparisons, ablations, and a deployable disaster-mask demo.

## Chosen direction

**Space Disaster Mapper** is a satellite-imagery segmentation project for flood and wildfire mapping. The strongest implementation path is to compare a lightweight U-Net/DeepLab-style baseline with NASA/IBM Prithvi-style foundation-model workflows on public flood/fire datasets.

Why this is the best resume project:

1. **Research-backed ML:** uses transformer/MAE geospatial foundation-model work rather than a toy classifier.
2. **Clear applied value:** floods and wildfires are real disaster-response use cases.
3. **Visual demo potential:** masks can be overlaid on satellite images in a web app.
4. **Quantitative evaluation:** IoU, Dice/F1, precision, recall, and per-class metrics are standard and recruiter-readable.
5. **Engineering depth:** includes data loading, model inference, API serving, frontend visualization, and reproducibility docs.

## Candidate comparison

| Direction | Main source | Strength | Weakness | Verdict |
| --- | --- | --- | --- | --- |
| **Prithvi + disaster mapping** | NASA/IBM Prithvi, Sen1Floods11, ImpactMesh | Strongest mix of research, datasets, metrics, demo, and job signal | Full fine-tuning can require GPU time | **Chosen** |
| SatMAE satellite pretraining | NeurIPS 2022 SatMAE | Strong research signal and self-supervised learning | More scene-classification oriented; heavier pretraining | Use as related work or optional baseline |
| Exoplanet detection | AstroNet / Google exoplanet-ml | Cool astronomy time-series project | Many clones exist; less visual demo value | Good fallback |
| Solar flare forecasting | GOES/SDO flare forecasting papers | Interesting space-weather prediction | Domain and labeling are messier | Good but harder to polish |

## Key research sources

### Prithvi-EO / Prithvi-100M

Prithvi-EO is a geospatial foundation model from IBM and NASA. The model card describes a temporal Vision Transformer trained with a masked-autoencoder objective on Harmonized Landsat Sentinel-2 data. It accepts remote-sensing data in video-like shape `(B, C, T, H, W)` and supports downstream tasks such as burn-scar segmentation, flood mapping, and land-cover classification.

Sources:

- Prithvi-EO model card: <https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M>
- Prithvi flood fine-tune: <https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11>
- Prithvi burn-scar fine-tune: <https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M-burn-scar>
- Fine-tuning code: <https://github.com/NASA-IMPACT/hls-foundation-os>
- Paper/preprint: Jakubik et al., "Foundation Models for Generalist Geospatial Artificial Intelligence," arXiv:2310.18660, <https://arxiv.org/abs/2310.18660>

### Sen1Floods11

Sen1Floods11 is a flood-mapping dataset for training and testing deep-learning flood algorithms with Sentinel imagery. It is useful for this project because it provides a known benchmark and labeled flood/water masks.

Sources:

- Paper: Bonafilia et al., "Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1," CVPR Workshops 2020, <https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.html>
- GitHub: <https://github.com/cloudtostreet/Sen1Floods11>

### ImpactMesh

ImpactMesh is a newer IBM/ESA/DLR multimodal and multitemporal disaster dataset for flood and wildfire mapping. It includes Sentinel-1 SAR, Sentinel-2 optical data, DEM information, and Copernicus EMS annotations. This is especially valuable for ablation studies such as optical-only vs SAR-only vs SAR+optical+DEM.

Sources:

- ImpactMesh GitHub: <https://github.com/IBM/ImpactMesh>
- ImpactMesh Flood: <https://huggingface.co/datasets/ibm-esa-geospatial/ImpactMesh-Flood>
- ImpactMesh Fire: <https://huggingface.co/datasets/ibm-esa-geospatial/ImpactMesh-Fire>
- IBM research article: <https://research.ibm.com/blog/geospatial-flood-and-wildfire-mapping>

### SatMAE

SatMAE is a NeurIPS 2022 masked-autoencoder method for temporal and multispectral satellite imagery. It is highly relevant related work, but less directly aligned with a flood/fire segmentation MVP than Prithvi because the official project focuses on fMoW-style satellite scene classification and requires more compute for serious pretraining.

Sources:

- Paper: <https://arxiv.org/abs/2207.08051>
- GitHub: <https://github.com/sustainlab-group/SatMAE>
- Project page: <https://sustainlab-group.github.io/SatMAE/>

### Exoplanet detection / AstroNet

AstroNet is a deep-learning system for identifying exoplanets in Kepler light curves. It is a strong space-science ML idea, but the project space is more saturated and less connected to a deployable computer-vision demo than disaster segmentation.

Sources:

- Paper: Shallue and Vanderburg, "Identifying Exoplanets with Deep Learning," <https://arxiv.org/abs/1712.05044>
- Google research code: <https://github.com/google-research/exoplanet-ml>

### Solar flare forecasting

Solar flare forecasting is another space-weather ML path using time-series and magnetogram data. It can be resume-worthy, but it usually needs more domain-specific metric handling such as TSS/HSS and careful rare-event evaluation.

Example source:

- Sun et al., "Predicting Solar Flares Using CNN and LSTM on Two Solar Cycles," <https://arxiv.org/abs/2204.03710>

## Original contribution framing

Safe and honest wording:

> Implemented and extended a Prithvi-inspired geospatial ML pipeline for satellite flood and wildfire segmentation, including baseline models, evaluation metrics, ablation comparisons, and a FastAPI + React visualization demo.

Avoid:

> Invented a new NASA flood-detection model.

## Suggested final resume bullet

Fill in actual metrics only after experiments:

> Built a paper-backed satellite disaster-mapping system using PyTorch, FastAPI, and React; implemented flood/wildfire segmentation baselines, evaluated IoU/Dice/precision/recall on Sen1Floods11 and ImpactMesh-style data, and deployed an interactive mask-visualization demo.
