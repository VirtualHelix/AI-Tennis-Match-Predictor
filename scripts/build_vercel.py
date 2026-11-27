#!/usr/bin/env python3
"""Verify the private model export and prepare its equivalent serving artifact."""
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.parse import urlsplit
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    result = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_export(archive_path, destination, expected, root=ROOT):
    """Extract only verified artifacts, never execute code from the archive."""
    archive_path, destination = Path(archive_path), Path(destination)
    if archive_path.stat().st_size != expected["archive_bytes"] or digest(archive_path) != expected["archive_sha256"]:
        raise ValueError("The model export does not match the reviewed release.")
    required = {"models/courtiq/model.joblib", "courtiq/model.py",
                "models/courtiq/validation_predictions.csv", "requirements-model.txt", "bundle_manifest.json"}
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if len(infos) != len(required) or {info.filename for info in infos} != required:
            raise ValueError("Unexpected members in the model export.")
        if any(info.is_dir() or info.file_size > 128 * 1024 * 1024 for info in infos):
            raise ValueError("Invalid artifact size or type.")
        manifest = json.loads(archive.read("bundle_manifest.json"))
        if manifest.get("format") != "courtiq-deployment-model" or manifest.get("version") != 2:
            raise ValueError("Unsupported model export format.")
        pins = dict(line.split("==", 1) for line in (root / "requirements/model.txt").read_text().splitlines()
                    if line and not line.startswith("#"))
        if manifest.get("packages") != pins:
            raise ValueError("The model export dependency pins differ from this release.")
        for package, version in pins.items():
            if metadata.version(package) != version:
                raise ValueError("The build installed the wrong version of " + package)
        targets = {"models/courtiq/model.joblib": ("source.joblib", expected["model_sha256"]),
                   "models/courtiq/validation_predictions.csv": ("validation_predictions.csv", expected["calibration_sha256"])}
        source = archive.read("courtiq/model.py")
        if sha256(source).hexdigest() != expected["source_sha256"] or source != (root / "courtiq/model.py").read_bytes():
            raise ValueError("The frozen model source does not match the release.")
        destination.mkdir(parents=True, exist_ok=True)
        for name, (filename, expected_hash) in targets.items():
            target = destination / filename
            with archive.open(name) as stream, target.open("xb") as output:
                shutil.copyfileobj(stream, output, 1024 * 1024)
            recorded = manifest.get("files", {}).get(name, {})
            if (target.stat().st_size != recorded.get("bytes") or digest(target) != expected_hash
                    or recorded.get("sha256") != expected_hash):
                raise ValueError("An exported artifact failed its integrity check.")
    return destination / "source.joblib", destination / "validation_predictions.csv"


def main():
    expected = json.loads((ROOT / "deployment/model.json").read_text())
    url = os.environ.get("COURTIQ_MODEL_BUNDLE_URL", "")
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or not (parsed.hostname or "").endswith(".private.blob.vercel-storage.com")
            or parsed.username or parsed.password or parsed.port not in (None, 443) or parsed.query or parsed.fragment):
        raise ValueError("Configure COURTIQ_MODEL_BUNDLE_URL with the private release artifact URL.")
    from vercel.blob import sync as blob
    from courtiq.deploy_model import pack_model

    output_directory = ROOT / "models/courtiq"
    output_directory.mkdir(parents=True, exist_ok=True)
    # Build beside the destination so the final rename is atomic even when
    # the system temporary directory is on another filesystem.
    with tempfile.TemporaryDirectory(prefix=".courtiq-build-", dir=output_directory) as temporary:
        temporary = Path(temporary)
        archive_path = temporary / "bundle.zip"
        count = 0
        with blob.stream_blob(url, access="private") as download, archive_path.open("xb") as output:
            for block in download:
                count += len(block)
                if count > expected["archive_bytes"]:
                    raise ValueError("The model download exceeds its reviewed size.")
                output.write(block)
        model, calibration = verify_export(archive_path, temporary / "verified", expected)
        packed = temporary / "deployment.joblib"
        report = pack_model(model, ROOT / "courtiq/model.py", packed)
        # Publish generated build files only after all checks and packing pass.
        os.replace(packed, output_directory / "deployment.joblib")
        shutil.copyfile(calibration, output_directory / "validation_predictions.csv")
        (output_directory / "deployment_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Verified model export and prepared serving artifact: {:.1f} MiB.".format(report["artifact_bytes"] / 1024**2))


if __name__ == "__main__":
    main()
