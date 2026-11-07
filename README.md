# RoadVision AI

Vehicle detection with YOLOv9. The original dataset exploration, cleaning, splitting, training, and plotting workflow is kept in one self-contained notebook, adapted for a Vast.ai GPU instance.

## Project structure

```text
notebooks/roadvision.ipynb  # Run this notebook
src/dataset.py             # Optional helper; the notebook does not import it
data/                     # Kaggle cache, cleaned dataset, prepared splits
models/yolov9-c.pt         # Downloaded pretrained weights
reports/<run_name>/        # Training plots, CSV, and weights/best.pt + last.pt
third_party/yolov9/        # Upstream YOLOv9 checkout (created by notebook)
requirements.txt
```

Generated datasets, weights, reports, and the upstream checkout are ignored by Git.

## Run on Vast.ai

1. Create a GPU instance with a **PyTorch + CUDA + Jupyter** template, preferably using Python 3.10 or 3.11. Allocate disk space for the dataset cache, cleaned copy, split copy, dependencies, and model checkpoints.
2. Open a terminal in Jupyter and clone this repository inside your instance's storage directory:

   ```bash
   git clone https://github.com/Oussamatiu/RoadVision-AI-.git
   cd RoadVision-AI-
   ```

   If already cloned, run `git pull` from the repository instead.
3. Open `notebooks/roadvision.ipynb` with the template's GPU Python kernel. Run the Vast.ai setup cells first. They find the project root, install project dependencies into the kernel environment, and check CUDA access. No Colab paths are required.
4. Run the remaining cells in order. Kaggle download may require authentication in your instance. YOLOv9 is cloned before use, and its separate dependencies are installed later in the notebook. If installation requests a kernel restart, restart before continuing.
5. Training defaults to **10 epochs, batch size 4, image size 640, GPU 0**. Edit `EPOCHS`, `BATCH_SIZE`, and `IMAGE_SIZE` in setup. Reduce batch size if GPU memory is insufficient.
6. Download `reports/<run_name>/weights/best.pt`, `last.pt`, and any plots/CSV you need before destroying the instance.

## Paths and reruns

All paths are resolved from the repository root. Launch the notebook with its working directory inside the repository. Dataset preparation uses `data/road_vehicle_clean` and `data/road_vehicle_split`. It refuses to overwrite an existing preparation directory; choose fresh paths in setup for a new preparation run.

To reuse prepared data after a kernel restart, rerun setup and imports, load the dataset configuration/class names, skip the cleaning and split-copy cells, then run YOLOv9 setup and the training-configuration cell. That cell rebuilds `DATA_YAML` using the existing split directories. Each training run gets a unique directory under `reports/`.

The notebook retains the supplied cleaning policy, including removing empty-label pairs from the clean copy. It preserves the source validation split and reserves approximately 15% of training candidates for testing with seed 42. Expected counts are calculated rather than fixed at the original dataset sizes. The final plots show training/validation results; held-out test evaluation is not performed.

## Validation

Notebook cell syntax, project-root discovery, project-relative paths, and construction of the GPU training command were checked without downloading the dataset or training. End-to-end training has not been run on Vast.ai. The template supplies PyTorch/CUDA; YOLOv9's upstream dependency and checkpoint compatibility depends on that runtime.
