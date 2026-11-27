#!/usr/bin/env python3
"""Package the saved CourtIQ model and its working dependency versions.

Run in the virtual environment that already runs the predictor. The exporter
reads only explicitly selected model artifacts; it does not unpickle, train,
change, publish, or upload a model. Both the current and legacy layouts work.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import zipfile


PACKAGES = ('numpy', 'pandas', 'scipy', 'scikit-learn', 'catboost', 'joblib', 'threadpoolctl')
MODEL_MEMBER = 'models/courtiq/model.joblib'
SOURCE_MEMBER = 'courtiq/model.py'
CALIBRATION_MEMBER = 'models/courtiq/validation_predictions.csv'
LEGACY_MODEL = 'models/research_v5/model.joblib'
LEGACY_SOURCE = 'src/v5/tennis_research_model.py'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _input_path(root, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _select_artifacts(root, model_path=None, source_path=None):
    current_model = _input_path(root, MODEL_MEMBER)
    legacy_model = _input_path(root, LEGACY_MODEL)
    if model_path is not None:
        model = _input_path(root, model_path)
    else:
        model = current_model if current_model.is_file() else legacy_model
    if not model.is_file():
        raise ValueError('Saved model not found. Expected ' + MODEL_MEMBER + ' or ' + LEGACY_MODEL +
                         '; use --model to select another file.')

    if source_path is not None:
        source = _input_path(root, source_path)
    elif model == legacy_model:
        # A user's older training source takes precedence over a newer checkout.
        legacy_source = _input_path(root, LEGACY_SOURCE)
        source = legacy_source if legacy_source.is_file() else _input_path(root, SOURCE_MEMBER)
    elif model == current_model:
        source = _input_path(root, SOURCE_MEMBER)
    else:
        raise ValueError('For a custom model path, also provide --model-code with its exact training source.')
    if not source.is_file():
        raise ValueError('Matching model source not found. Use --model-code with the exact training script.')

    artifacts = {MODEL_MEMBER: model, SOURCE_MEMBER: source}
    calibration = model.parent / 'validation_predictions.csv'
    if calibration.is_file():
        artifacts[CALIBRATION_MEMBER] = calibration
    return artifacts


def _provenance(path, root):
    try:
        relative_path = path.relative_to(root).as_posix()
    except ValueError:
        relative_path = None
    return {'source_relative_path': relative_path, 'source_filename': path.name,
            'external_source': relative_path is None}


def export_bundle(root, destination, package_versions=None, model_path=None, source_path=None):
    root = Path(root).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise ValueError('The output already exists. Use --out with a new filename to keep the previous bundle.')
    artifacts = _select_artifacts(root, model_path=model_path, source_path=source_path)
    versions = {}
    for package in PACKAGES:
        try:
            version = (package_versions[package] if package_versions is not None
                       else metadata.version(package))
        except (metadata.PackageNotFoundError, KeyError):
            raise ValueError('Activate the virtual environment that runs CourtIQ; missing package: ' + package)
        if not isinstance(version, str) or not re.fullmatch(r'[A-Za-z0-9.!+_-]+', version):
            raise ValueError('Could not capture a safe version pin for ' + package)
        versions[package] = version
    manifest = {
        'format': 'courtiq-deployment-model',
        'version': 2,
        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'python_version': platform.python_version(),
        'packages': versions,
        'files': {},
        'calibration_included': CALIBRATION_MEMBER in artifacts,
    }
    for name, path in artifacts.items():
        if path.stat().st_size == 0:
            raise ValueError('Artifact is empty: ' + name)
        manifest['files'][name] = {'bytes': path.stat().st_size, 'sha256': sha256(path),
                                  **_provenance(path, root)}

    destination.parent.mkdir(parents=True, exist_ok=True)
    created_output = False
    try:
        with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            created_output = True
            for name, path in artifacts.items():
                archive.write(path, arcname=name)
            archive.writestr('requirements-model.txt', ''.join(name + '==' + versions[name] + '\n'
                                                              for name in PACKAGES))
            archive.writestr('bundle_manifest.json', json.dumps(manifest, indent=2) + '\n')
        # Check source bytes and archive integrity without loading the model.
        with zipfile.ZipFile(destination) as archive:
            if archive.testzip() is not None:
                raise ValueError('The archive did not pass its integrity check.')
            for name in artifacts:
                digest = hashlib.sha256()
                with archive.open(name) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(block)
                if digest.hexdigest() != manifest['files'][name]['sha256']:
                    raise ValueError('An artifact changed while being copied. Stop training and retry: ' + name)
    except Exception:
        if created_output:
            destination.unlink(missing_ok=True)
        raise
    return destination, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.', help='Existing AI-Tennis-Match-Predictor directory.')
    parser.add_argument('--model', help='Saved model path, relative to --root or absolute.')
    parser.add_argument('--model-code', '--source', dest='source_path',
                        help='Exact training source, relative to --root or absolute.')
    parser.add_argument('--out', default=str(Path.home() / 'Downloads' / 'courtiq_model_bundle.zip'),
                        help='New output archive; existing files are preserved.')
    args = parser.parse_args()
    try:
        destination, manifest = export_bundle(args.root, args.out, model_path=args.model,
                                              source_path=args.source_path)
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        parser.exit(1, 'Could not prepare deployment: ' + str(error) + '\n')
    print('Created: ' + str(destination))
    print('Bundle size: {:.1f} MB'.format(destination.stat().st_size / (1024 * 1024)))
    print('Included: saved model, selected training source, package versions' +
          (', and calibration results.' if manifest['calibration_included'] else '.'))
    if not manifest['calibration_included']:
        print('Calibration CSV was not found; the calibration chart will be unavailable.')
    print('The model and source have not been loaded; deployment must verify their compatibility.')


if __name__ == '__main__':
    main()
