import io
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from src.inference import detect_image


class InferenceTest(unittest.TestCase):
    def test_subprocess_result_and_cleanup(self):
        # A tiny stand-in checks the CLI/file boundary without GPU/model weights.
        with tempfile.TemporaryDirectory(prefix='model with spaces ') as directory:
            root = Path(directory)
            weights = root / 'best.pt'
            weights.touch()
            script = root / 'detect_dual.py'
            script.write_text('''import sys, shutil
from pathlib import Path
a=sys.argv
def arg(name): return a[a.index(name)+1]
source=Path(arg('--source'))
out=Path(arg('--project'))/arg('--name')
(out/'labels').mkdir(parents=True)
shutil.copy2(source,out/'input.jpg')
(out/'labels/input.txt').write_text('0 0.5 0.5 0.2 0.2 0.9\\n')
(Path(__file__).parent/'temporary-path.txt').write_text(str(source.parent))
''')
            image = io.BytesIO()
            Image.new('RGB', (20, 20)).save(image, format='PNG')
            annotated, labels = detect_image(image.getvalue(), weights, root, device='cpu')
            with Image.open(io.BytesIO(annotated)) as output:
                self.assertEqual(output.size, (20, 20))
            self.assertIn('0.9', labels)
            self.assertFalse(Path((root/'temporary-path.txt').read_text()).exists())
            script.write_text('raise RuntimeError("fixture failure")')
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                detect_image(image.getvalue(), weights, root)
            script.write_text('pass')
            with self.assertRaisesRegex(RuntimeError, 'did not produce'):
                detect_image(image.getvalue(), weights, root)

    def test_missing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                detect_image(b'', Path(directory)/'missing.pt', directory)


if __name__ == '__main__':
    unittest.main()
