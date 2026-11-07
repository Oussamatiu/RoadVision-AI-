"""Run original dual-head YOLOv9 checkpoints through upstream detect_dual.py."""
import io
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image, ImageOps


def detect_image(image_bytes, weights, yolo_dir, confidence=0.25, device=''):
    weights, yolo_dir = Path(weights).resolve(), Path(yolo_dir).resolve()
    if not weights.is_file():
        raise FileNotFoundError('Trained checkpoint not found. Export best.pt from the notebook first.')
    if not (yolo_dir / 'detect_dual.py').is_file():
        raise FileNotFoundError('YOLOv9 is missing. Run the notebook YOLOv9 setup cells first.')
    if not 0 < confidence <= 1:
        raise ValueError('Confidence must be between zero and one.')
    if device not in {'', '0', 'cpu'}:
        raise ValueError('Choose automatic, GPU 0, or CPU.')
    with tempfile.TemporaryDirectory(prefix='roadvision-') as directory:
        work = Path(directory)
        source = work / 'input.jpg'
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            image.convert('RGB').save(source, quality=95)
        command = [
            sys.executable, str(yolo_dir / 'detect_dual.py'),
            '--weights', str(weights), '--source', str(source),
            '--imgsz', '640', '--conf-thres', str(confidence),
            '--device', device, '--project', str(work / 'results'),
            '--name', 'prediction', '--exist-ok', '--save-txt', '--save-conf',
        ]
        try:
            result = subprocess.run(command, cwd=yolo_dir, capture_output=True,
                                    text=True, timeout=300, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError('YOLOv9 inference failed.\n' + (exc.stderr or exc.stdout or '')[-6000:]) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('Inference exceeded five minutes. Try a GPU instance.') from exc
        output = work / 'results/prediction/input.jpg'
        if not output.is_file():
            raise RuntimeError('YOLOv9 did not produce an image.\n' + result.stderr[-2000:])
        label_path = output.parent / 'labels/input.txt'
        labels = label_path.read_text() if label_path.exists() else ''
        return output.read_bytes(), labels
