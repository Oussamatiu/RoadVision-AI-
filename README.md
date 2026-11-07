# RoadVision AI

Vehicle detection with YOLOv9 using the Kaggle road vehicle images dataset.

## Notebook

[Open notebooks/roadvision.ipynb](notebooks/roadvision.ipynb)

The notebook contains the original supplied code, unchanged and in its original order, divided into 48 code cells with section headings. It is self-contained and does not import the separate `src/dataset.py` helper.

Sections cover dataset download and exploration, annotation inspection, cleaning, class distribution, train/validation/test splits, YOLOv9 setup, training, and result plots.

## Usage

Open the notebook in Google Colab or Jupyter. The supplied code uses Colab-style `/content` paths and GPU device 0. Install pandas, matplotlib, Pillow, PyYAML, kagglehub, and scikit-learn before running the first cells. YOLOv9 dependencies are installed later in the notebook.

The original code's fixed dataset counts, file-removal operations on the clean copy, and commands are preserved. The YOLOv9 setup changes into `/content/yolov9` before cloning; on a fresh runtime, clone into that location first and then change into it. Review the cells before execution. The notebook has been checked for cell syntax and exact source preservation, but has not been executed end to end.

## Simple project structure

```text
notebooks/roadvision.ipynb  # Original code organized into cells
src/dataset.py             # Separate helper from the initial setup; unused by this notebook
data/                     # Space for datasets
models/                   # Space for model weights
reports/                  # Space for reports
requirements.txt          # Initial project dependencies; also install scikit-learn for this notebook
```

The original notebook writes to its `/content` paths rather than these placeholder project directories. No dataset or trained model is included.
