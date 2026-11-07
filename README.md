# RoadVision AI

Vehicle detection with YOLOv9, using the Kaggle road vehicle images dataset.
This project organizes the supplied exploration, cleaning, splitting, and training code into one notebook and a small helper module.

## Project structure

```text
RoadVision-AI-/
├── notebooks/
│   └── roadvision.ipynb   # Step-by-step workflow
├── src/
│   └── dataset.py        # Label validation and dataset preparation
├── data/                 # Generated datasets (ignored by Git)
├── models/               # Downloaded weights (ignored by Git)
├── reports/              # Training runs and plots (ignored by Git)
├── requirements.txt
└── README.md
```

## Run locally

Use Python 3.10 or 3.11. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter lab notebooks/roadvision.ipynb
```

Run the notebook sections in order. Dataset download requires internet access and may require Kaggle authentication. The notebook also works in Google Colab: run its first setup cell to clone the project, then install the dependencies in the next cell.

## Notebook workflow

1. Set up paths and dependencies.
2. Download and inspect the dataset and its class configuration.
3. Check labels, bounding boxes, and image readability.
4. Plot class counts and copy valid pairs into train/validation/test folders.
5. Set up YOLOv9 and pretrained weights.
6. Train and view the resulting plots.

The original downloaded dataset is never modified. Missing or invalid labels and unreadable images are excluded from the prepared copy. Empty annotations are retained as valid background examples by default; set `DROP_EMPTY = True` to reproduce the original removal policy. Validation and existing test splits are preserved. If no test split exists, 15% of valid training candidates are reserved using seed 42. Counts are computed from the current dataset rather than hard-coded.

Choose a fresh output directory when preparing another dataset; existing output folders are rejected to avoid mixing runs. Sample-level splitting does not guarantee scene-level separation for related frames; review the dataset before interpreting test performance.

Training is optional, requires a suitable CUDA GPU, and defaults to 10 epochs, batch size 4, and image size 640. YOLOv9 and its separate dependencies are installed by the notebook into `third_party/yolov9`. Upstream dependency compatibility depends on the runtime. No dataset, trained model, metrics, or training results are included in this repository.
