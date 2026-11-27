"""Model-export boundaries; artifact bytes deliberately are not pickle files."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile


SOURCE = Path(__file__).resolve().parents[1] / 'scripts/export_model.py'
SPEC = importlib.util.spec_from_file_location('courtiq_model_exporter', SOURCE)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)
VERSIONS = {name: '1.2.3' for name in exporter.PACKAGES}


class ExportModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='courtiq-export-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'export.zip'

    def write(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def current_layout(self):
        self.write(exporter.MODEL_MEMBER, b'opaque saved-model bytes')
        self.write(exporter.SOURCE_MEMBER, b'# exact model source\n')

    def test_current_layout_allowlist_provenance_and_python39_syntax(self):
        self.current_layout()
        self.write(exporter.CALIBRATION_MEMBER, b'probability_a,outcome_a\n0.6,1\n')
        self.write('.env', b'SECRET=must-not-be-exported')
        self.write('data/raw/private.csv', b'not an allowed artifact')
        output, manifest = exporter.export_bundle(self.root, self.output, VERSIONS)
        with zipfile.ZipFile(output) as archive:
            self.assertEqual(set(archive.namelist()), {exporter.MODEL_MEMBER, exporter.SOURCE_MEMBER,
                                                     exporter.CALIBRATION_MEMBER,
                                                     'requirements-model.txt', 'bundle_manifest.json'})
            self.assertIsNone(archive.testzip())
            for name, details in manifest['files'].items():
                self.assertEqual(details['source_relative_path'], name)
                self.assertFalse(details['external_source'])
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), details['sha256'])
            self.assertEqual(json.loads(archive.read('bundle_manifest.json')), manifest)
        ast.parse(SOURCE.read_text(), feature_version=(3, 9))

    def test_legacy_layout_keeps_legacy_source_even_if_newer_package_exists(self):
        self.write(exporter.LEGACY_MODEL, b'legacy model')
        self.write(exporter.LEGACY_SOURCE, b'# legacy training source\n')
        self.write(exporter.SOURCE_MEMBER, b'# different newer source\n')
        self.write('models/research_v5/validation_predictions.csv', b'legacy calibration')
        output, manifest = exporter.export_bundle(self.root, self.output, VERSIONS)
        with zipfile.ZipFile(output) as archive:
            self.assertEqual(archive.read(exporter.SOURCE_MEMBER), b'# legacy training source\n')
            self.assertEqual(archive.read(exporter.MODEL_MEMBER), b'legacy model')
            self.assertEqual(archive.read(exporter.CALIBRATION_MEMBER), b'legacy calibration')
        self.assertEqual(manifest['files'][exporter.MODEL_MEMBER]['source_relative_path'], exporter.LEGACY_MODEL)
        self.assertEqual(manifest['files'][exporter.SOURCE_MEMBER]['source_relative_path'], exporter.LEGACY_SOURCE)

    def test_explicit_paths_override_defaults_and_require_an_explicit_custom_source(self):
        self.current_layout()
        self.write('experiment/saved.joblib', b'custom model')
        self.write('experiment/trainer.py', b'# custom source\n')
        with self.assertRaisesRegex(ValueError, 'also provide --model-code'):
            exporter.export_bundle(self.root, self.output, VERSIONS, model_path='experiment/saved.joblib')
        output, manifest = exporter.export_bundle(self.root, self.output, VERSIONS,
                                                  model_path='experiment/saved.joblib',
                                                  source_path='experiment/trainer.py')
        with zipfile.ZipFile(output) as archive:
            self.assertEqual(archive.read(exporter.MODEL_MEMBER), b'custom model')
            self.assertEqual(archive.read(exporter.SOURCE_MEMBER), b'# custom source\n')
        self.assertEqual(manifest['files'][exporter.SOURCE_MEMBER]['source_relative_path'], 'experiment/trainer.py')

    def test_existing_output_is_preserved(self):
        self.current_layout()
        self.output.write_bytes(b'existing output')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            exporter.export_bundle(self.root, self.output, VERSIONS)
        self.assertEqual(self.output.read_bytes(), b'existing output')

    def test_exclusive_creation_race_does_not_delete_someone_elses_file(self):
        self.current_layout()
        original = zipfile.ZipFile

        def race(filename, mode='r', *args, **kwargs):
            if mode == 'x':
                Path(filename).write_bytes(b'created by another process')
            return original(filename, mode, *args, **kwargs)

        with patch.object(exporter.zipfile, 'ZipFile', side_effect=race):
            with self.assertRaises(FileExistsError):
                exporter.export_bundle(self.root, self.output, VERSIONS)
        self.assertEqual(self.output.read_bytes(), b'created by another process')

    def test_changed_artifact_is_detected_and_incomplete_output_removed(self):
        self.current_layout()
        original = zipfile.ZipFile.write

        def change_before_write(archive, filename, arcname=None, *args, **kwargs):
            if arcname == exporter.SOURCE_MEMBER:
                Path(filename).write_bytes(b'# changed while exporting\n')
            return original(archive, filename, arcname, *args, **kwargs)

        with patch.object(zipfile.ZipFile, 'write', change_before_write):
            with self.assertRaisesRegex(ValueError, 'changed while being copied'):
                exporter.export_bundle(self.root, self.output, VERSIONS)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
