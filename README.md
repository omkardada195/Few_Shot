# Few_Shot

Few-shot medical image segmentation coursework repository for EMS741.

## Project goal
This project compares:
- a baseline segmentation model trained from scratch on a few annotated images
- a Reptile-based few-shot meta-learning model
- performance under 1-shot, 3-shot, and 5-shot adaptation

## Repository structure
- `notebooks/` : main Google Colab notebook
- `src/` : helper Python modules
- `results/` : saved outputs, plots, and tables

## Dataset
The dataset is **not stored in this repository**.
In Google Colab, the dataset should be loaded from Google Drive.

Expected dataset structure:

ems741_cw_data/
- train/
  - task_2/
  - task_3/
  - task_5/
  - task_7/
- val/
  - task_4/
  - task_6/
- test/
  - task_1/
  - task_8/

Each task folder must contain:
- `images/`
- `masks/`

## Colab usage
1. Clone this repository
2. Mount Google Drive
3. Point the notebook to the dataset root
4. Run all cells

## Notes
- Images and masks are resized to 192x192
- Empty masks are not removed from evaluation
- Support selection may prefer positive masks for more informative few-shot adaptation