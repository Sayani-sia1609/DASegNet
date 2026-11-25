# Degradation-Aware Semantic Segmentation — EDA modules

This folder contains the Python modules for dataset loading, exploratory data analysis, semantic class distribution analysis, image visualization, and controlled degradation generation.

## Project files

- `data_loading.py`: ACDC/Cityscapes-C archive connections, RGB-mask matching, sampling, and streaming.
- `dataset_analysis.py`: image-quality measurements and end-to-end dataset EDA.
- `class_distribution.py`: semantic mask selection and pixel-level class distribution.
- `image_visualization.py`: reusable plots and image preview grids.
- `preprocessing.py`: resizing/normalization, controlled corruptions, PSNR/SSIM, and degradation experiment export.
- `run_pipeline.py`: runs the complete workflow by calling the three main stages in sequence.

## Run the project in Google Colab

Follow these steps in order. You only need to run `run_pipeline.py`; do not run the individual module files separately.

### Step 1 — Upload the ZIP file

Download `degradation_aware_eda_modules.zip` to your computer, open a new Google Colab notebook, and run:

```python
from google.colab import files

uploaded = files.upload()
```

Select `degradation_aware_eda_modules.zip` when the file picker opens.

### Step 2 — Extract the project files

Run this cell:

```python
import zipfile
import os

zip_path = "/content/degradation_aware_eda_modules.zip"
extract_path = "/content"

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

os.chdir("/content/degradation_aware_eda")

print("Project files:")
print(os.listdir())
```

After extraction, the project files should be in `/content/degradation_aware_eda/`.

### Step 3 — Install the dependencies

Run:

```python
!pip install remotezip huggingface_hub requests pandas numpy matplotlib seaborn pillow opencv-python tqdm scikit-image
```

Wait for the installation to finish before continuing.

### Step 4 — Mount Google Drive

The pipeline saves its results under your Google Drive, so mount it:

```python
from google.colab import drive

drive.mount('/content/drive')
```

Follow the authorization prompt.

### Step 5 — Run the complete pipeline

Make sure the current working directory is `/content/degradation_aware_eda` (the extraction cell above sets it), then run:

```python
!python run_pipeline.py
```

`run_pipeline.py` runs these stages in order:

1. **Dataset loading and EDA** — calls `run_dataset_analysis()` from `dataset_analysis.py`.
2. **Semantic class distribution** — calls `save_class_distribution()` from `class_distribution.py`.
3. **Controlled degradation generation** — calls `generate_degradation_dataset()` from `preprocessing.py`, with `max_images=10`.

The other modules are imported and used by these stages as needed. Keep all the `.py` files together in the same folder so the imports work correctly.

## Output location

The pipeline is configured to save results under:

```text
/content/drive/MyDrive/Cityscapes_EDA/results/
```

This includes the `csv/`, `graphs/`, and `reports/` output folders, as well as:

```text
/content/drive/MyDrive/Cityscapes_EDA/results/degradation_experiments/
```

The degradation experiment folder contains the generated degraded samples and related metrics/summaries, as configured by the pipeline.

## Important notes

- The pipeline uses the Google Drive output path configured in `run_pipeline.py`. If you want a different destination, edit that path in `run_pipeline.py` before running.
- The first run may take time because the datasets are accessed and analysed.
- ACDC RGB entries are restricted to fog train/validation images, matching the current module configuration.
- TrainIds annotation masks are preferred; labelIds are used as a fallback.
- Cityscapes-C is streamed from its archive rather than fully extracted.
- The default degradation experiment uses up to 10 sampled ACDC images, 11 corruption types, and 5 severity levels (up to 550 variants if all source images and variants are available).
- Controlled degradations modify RGB images only; semantic masks are not modified. For later segmentation training, apply geometric transforms identically to RGB images and masks, and keep training/validation splits separate to avoid data leakage.
- `preprocessing.py` expects RGB `uint8` NumPy arrays.

## Running again

If the Colab runtime disconnects or resets, upload and extract the ZIP again, reinstall dependencies if needed, remount Google Drive, and run `!python run_pipeline.py` from the project directory.
