# RoadVision AI

Vehicle detection with YOLOv9. The original dataset exploration, cleaning, splitting, training, and plotting workflow is kept in one self-contained notebook, adapted for a Vast.ai GPU instance.

## Project structure

```text
app.py                    # Streamlit image detection app
requirements-streamlit.txt # Streamlit dependencies
notebooks/roadvision.ipynb  # Train and export your model
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

## Streamlit: use the trained model

1. Complete notebook training and run the final **Export your trained model for Streamlit** cell. This copies your trained checkpoint to `models/best.pt`. The app also finds checkpoints in `reports/*/weights/best.pt`.
2. In a terminal, activate the same Python environment used by the notebook and run from the project root:

   ```bash
   python -m pip install -r requirements-streamlit.txt
   python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
   ```

3. For Vast.ai access, configure a port mapping for container TCP port **8501** when creating the instance. Open `http://<instance-public-IP>:<mapped-external-port>` using the actual mapping shown by Vast.ai; the external port may differ from 8501. Alternatively, use an SSH tunnel through the instance's SSH connection and bind Streamlit to `127.0.0.1`. Keep access private if you do not intend to share the app; Streamlit itself does not add login authentication.
4. Select the trained model, upload a JPG/PNG/WebP image, choose confidence, and click **Detect vehicles**. Download the annotated image or the YOLO-format detection text.

On a different machine, copy `models/best.pt`, clone YOLOv9 into `third_party/yolov9`, and install its requirements plus `requirements-streamlit.txt` in a compatible PyTorch environment. Use checkpoints you trained or trust. This app supports the original dual-head checkpoints from `train_dual.py`, not converted or arbitrary YOLO models. It does not use the COCO pretrained weights as a substitute for your trained vehicle model.

Inference uses upstream `detect_dual.py` in a separate process for each request. The model reloads each time, so this simple version is intended for individual images rather than real-time video. Uploaded images and inference files are processed in a temporary directory and removed afterward; results remain in the browser session for downloading. Large images are limited to 20 MB.

The app's UI and inference wrapper can be checked without model weights; real vehicle predictions require a completed training checkpoint. No trained weights are committed to GitHub.
