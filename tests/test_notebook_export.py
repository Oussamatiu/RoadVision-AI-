"""Exercise notebook checkpoint handoff without loading a model or using a GPU."""
import ast
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import yaml

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/roadvision.ipynb'


class NotebookExportTest(unittest.TestCase):
    def test_export_and_archive(self):
        nb = json.loads(NOTEBOOK.read_text())
        cells = {c['id']: ''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code'}
        for source in cells.values():
            ast.parse(source)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports, models, yolo = [root / name for name in ['reports', 'models', 'yolo']]
            for folder in [reports, models, yolo]:
                folder.mkdir()
            run = reports / 'fixture_run'
            (run / 'weights').mkdir(parents=True)
            (run / 'weights/best.pt').write_bytes(b'fixture best checkpoint')
            (run / 'weights/last.pt').write_bytes(b'fixture last checkpoint')
            config = root / 'data.yaml'
            config.write_text(yaml.safe_dump({'names': ['car'], 'test': str(root/'test')}))
            # Recover an older run through YOLOv9's opt.yaml, without live notebook state.
            (run / 'opt.yaml').write_text(yaml.safe_dump({'data': str(config)}))
            ns = dict(ROOT=root, REPORTS_DIR=reports, MODELS_DIR=models, YOLO_DIR=yolo)
            exec(cells['select-saved-run'], ns)
            def command(args, **kwargs):
                return subprocess.CompletedProcess(args, 0, stdout='fixture-revision\n' if args[0]=='git' else 'fixture-package==1\n')
            with patch('subprocess.run', side_effect=command):
                exec(cells['save-model-bundle'], ns)
            self.assertEqual((models/'best.pt').read_bytes(), (run/'weights/best.pt').read_bytes())
            bundle = ns['BUNDLE_DIR']
            metadata = json.loads((bundle/'model_info.json').read_text())
            self.assertEqual(metadata['class_names'], {'0': 'car'})
            self.assertEqual(metadata['prediction_check'], 'not run by this export')
            self.assertEqual(metadata['sha256']['best.pt'], ns['sha256_file'](models/'best.pt'))
            exec(cells['package-model'], ns)
            import zipfile
            with zipfile.ZipFile(ns['archive']) as zipped:
                self.assertTrue({'best.pt', 'last.pt', 'classes.json', 'model_info.json', 'environment.txt'} <= set(zipped.namelist()))
                self.assertEqual(zipped.read('best.pt'), (models/'best.pt').read_bytes())
            (run/'weights/best.pt').write_bytes(b'')
            with self.assertRaises(FileNotFoundError):
                exec(cells['select-saved-run'], ns)


if __name__ == '__main__':
    unittest.main()
