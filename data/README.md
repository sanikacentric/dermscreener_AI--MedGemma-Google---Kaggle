# ISIC Dataset Instructions

In order to run `baseline_inference.py` or `finetune.py`, you must download the ISIC 2019 dataset.

## Download
1. Go to [ISIC Challenge 2019](https://challenge.isic-archive.com/data/)
2. Download the Training Images (`ISIC_2019_Training_Input.zip`)
3. Download the Training Ground Truth (`ISIC_2019_Training_GroundTruth.csv`)

## Processing
Extract the zip file to `data/processed/`. 
Rename the Ground Truth CSV to `metadata.csv` and ensure it has columns:
- `image_path` (e.g., `ISIC_0000000.jpg`)
- `label` (e.g., `melanoma`)

For the `label` column, convert one-hot encoded columns from the original ISIC format into a single categorical text column mapping to the 8 standard conditions.
