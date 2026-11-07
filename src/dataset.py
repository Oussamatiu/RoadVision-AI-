"""Small helpers for inspecting and preparing YOLO detection datasets."""
from collections import Counter
import math
from pathlib import Path
import random
import shutil

from PIL import Image
import yaml

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def find_label(image_path):
    parts = list(Path(image_path).parts)
    positions = [i for i, part in enumerate(parts) if part.lower() == 'images']
    if not positions:
        raise ValueError(f"No images directory in {image_path}")
    parts[positions[-1]] = 'labels'
    return Path(*parts).with_suffix('.txt')


def load_classes(yaml_path):
    config = yaml.safe_load(Path(yaml_path).read_text())
    names = config['names']
    names = ({int(k): str(v) for k, v in names.items()} if isinstance(names, dict)
             else dict(enumerate(map(str, names))))
    if not names or set(names) != set(range(len(names))):
        raise ValueError('Class IDs must be consecutive, starting at zero.')
    return names


def inspect_dataset(root, class_names):
    """Return one record per image. Empty labels are valid background images."""
    records = []
    for image in sorted(Path(root).rglob('*')):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        errors, classes = [], []
        label = find_label(image)
        try:
            with Image.open(image) as img:
                img.verify()
            with Image.open(image) as img:
                img.load()
        except Exception as exc:
            errors.append(f'Unreadable image: {exc}')
        empty = False
        if not label.is_file():
            errors.append('Missing label')
        else:
            try:
                lines = label.read_text().splitlines()
                empty = not any(line.strip() for line in lines)
                for number, line in enumerate(lines, 1):
                    if not line.strip():
                        continue
                    try:
                        values = list(map(float, line.split()))
                        if len(values) != 5 or not all(map(math.isfinite, values)):
                            raise ValueError('Expected five finite numbers')
                        cls, x, y, w, h = values
                        if not cls.is_integer() or int(cls) not in class_names:
                            raise ValueError('Unknown class ID')
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                            raise ValueError('Invalid normalized box')
                        if x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6:
                            raise ValueError('Box outside image')
                        classes.append(int(cls))
                    except ValueError as exc:
                        errors.append(f'Line {number}: {exc}')
            except UnicodeError as exc:
                errors.append(f'Unreadable label: {exc}')
        records.append(dict(image=image, label=label, empty=empty,
                            classes=classes, errors=errors))
    return records


def source_split(image):
    """Support both train/images/a.jpg and images/train/a.jpg."""
    parts = Path(image).parts
    index = max(i for i, part in enumerate(parts) if part.lower() == 'images')
    candidates = parts[max(0, index-1):index] + parts[index+1:-1]
    for name in candidates:
        if name.lower() in {'train', 'valid', 'val', 'test'}:
            return 'valid' if name.lower() == 'val' else name.lower()
    raise ValueError(f'Cannot determine source split: {image}')


def prepare_dataset(records, destination, class_names, test_fraction=0.15,
                    seed=42, drop_empty=False):
    """Copy valid pairs to a NEW directory; preserve validation and existing test."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f'{destination} already exists. Choose a fresh output folder.')
    if not 0 < test_fraction < 1:
        raise ValueError('test_fraction must be between zero and one')
    splits = {'train': [], 'valid': [], 'test': []}
    for record in records:
        if record['errors'] or (drop_empty and record['empty']):
            continue
        splits[source_split(record['image'])].append(record)
    if not splits['train'] or not splits['valid']:
        raise ValueError('Both training and validation images are required.')
    if not splits['test']:
        if len(splits['train']) < 2:
            raise ValueError('At least two training images are required to create a test split.')
        random.Random(seed).shuffle(splits['train'])
        size = min(len(splits['train'])-1, max(1, round(len(splits['train'])*test_fraction)))
        splits['test'] = splits['train'][:size]
        splits['train'] = splits['train'][size:]
    # Unique output names prevent collisions across nested source directories.
    for split, items in splits.items():
        image_dir = destination / 'images' / split
        label_dir = destination / 'labels' / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for index, record in enumerate(items):
            name = f'{index:06d}_{record["image"].stem}'
            shutil.copy2(record['image'], image_dir / (name + record['image'].suffix))
            shutil.copy2(record['label'], label_dir / (name + '.txt'))
    config = {split: str(destination / 'images' / folder)
              for split, folder in [('train', 'train'), ('val', 'valid'), ('test', 'test')]}
    config.update(nc=len(class_names), names=[class_names[i] for i in range(len(class_names))])
    data_yaml = destination / 'data.yaml'
    data_yaml.write_text(yaml.safe_dump(config, sort_keys=False))
    return data_yaml, {split: len(items) for split, items in splits.items()}


def class_counts(records):
    return Counter(cls for record in records if not record['errors'] for cls in record['classes'])
