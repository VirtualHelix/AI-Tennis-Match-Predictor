"""Build a smaller serving artifact without changing the frozen model math.

The training bundle keeps every observation in three separate historical stores.
The Sipko and momentum stores have finite query windows. Once a snapshot is
released, observations older than their earliest possible query window cannot
contribute to predictions at or after that snapshot. Earlier predictions replay
``bundle['history']`` from scratch, which this packer preserves in full.

This is an offline packaging step. It never retrains or changes fitted weights.
The existing ``courtiq.model.TennisPredictor`` loads the resulting joblib file.
"""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path


FORMAT = 'courtiq-serving-snapshot'
VERSION = 1
SUPPORTED_SOURCE_SHA256 = '589fcf1b9426843bcd6995f390aa10bebd9ff169a635046d6eb91024247ccda5'
DAY = 86400.0
# These are exactly the released-record fields read by _player_context,
# matchup, CourtIQService._prepare_players, and CourtIQService._stats.
# Full original source rows remain in bundle['history'] for H2H/recent results
# and for the frozen predictor's historical replay path.
CONTEXT_FIELDS = (
    'date', 'known_at', 'surface', 'opponent', 'win', 'completed',
    'match_id', 'event_key', 'retirement_proxy', 'minutes',
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _window_store(store, cutoff, date_of, known_of):
    """Keep original observations/order and their aligned known-at timestamps."""
    result = dict(store)
    result['records'] = defaultdict(list)
    result['times'] = defaultdict(list)
    for player, rows in store['records'].items():
        times = store['times'].get(player, ())
        if len(times) != len(rows):
            raise ValueError('Historical records and timestamps are misaligned.')
        retained = []
        retained_times = []
        for stamp, row in zip(times, rows):
            if stamp != known_of(row):
                raise ValueError('Historical record timestamp does not match its index.')
            if date_of(row) >= cutoff:
                retained.append(row)
                retained_times.append(stamp)
        result['records'][player] = retained
        result['times'][player] = retained_times
    result['seen'] = {key: value for key, value in store['seen'].items()
                      if date_of(value) >= cutoff}
    return result


def compact_bundle(bundle, module):
    """Return a new outer bundle containing exactly equivalent serving state.

    Mutable dictionaries/lists are rebuilt; retained numeric observations are
    not modified. An unreleased pending result is rejected because it would
    require a more general proof about future update and duplicate handling.
    """
    source_hash = file_sha256(module.__file__)
    if source_hash != SUPPORTED_SOURCE_SHA256:
        raise ValueError('This packer only supports its reviewed frozen feature implementation.')
    if bundle.get('source_sha256') != source_hash:
        raise ValueError('The bundle must match its exact frozen training source.')
    if bundle.get('deployment_compaction') is not None:
        raise ValueError('This artifact has already been compacted.')
    original = bundle['engine_state']
    if original['pending']:
        raise ValueError('Deployment packing requires a fully released snapshot.')
    now = module._sf_stamp(original['current_date'])
    if now is None or not math.isfinite(now):
        raise ValueError('Snapshot date is invalid.')
    sipko = original['sipko']
    momentum = original['momentum']
    for value in (sipko['refresh'], sipko['history_days'], momentum['window']):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Feature windows must be positive and finite.')
    # Surface correlations use the *start* of a fixed refresh bucket, which can
    # precede the snapshot. Retain its complete history window as well.
    bucket_start = math.floor(now / sipko['refresh']) * sipko['refresh']
    sipko_cutoff = bucket_start - sipko['history_days'] * DAY
    momentum_cutoff = now - momentum['window'] * DAY
    new_sipko = _window_store(sipko, sipko_cutoff, lambda r: r['date'], lambda r: r['known'])
    new_sipko.update(_profile_cache={}, _profile_time=None, _surface_cache={})
    new_momentum = _window_store(momentum, momentum_cutoff, lambda r: r[0], lambda r: r[1])
    history = {}
    for player, rows in original['history'].items():
        history[player] = [{field: row[field] for field in CONTEXT_FIELDS} for row in rows]
    state = dict(original, sipko=new_sipko, momentum=new_momentum, history=history)
    result = dict(bundle, engine_state=state)
    count = lambda store: sum(len(rows) for rows in store['records'].values())
    result['deployment_compaction'] = {
        'format': FORMAT, 'version': VERSION,
        'snapshot_date': str(original['current_date']),
        'sipko_cutoff_timestamp': sipko_cutoff,
        'momentum_cutoff_timestamp': momentum_cutoff,
        'sipko_observations_before': count(sipko),
        'sipko_observations_after': count(new_sipko),
        'momentum_observations_before': count(momentum),
        'momentum_observations_after': count(new_momentum),
        'context_history_rows': sum(len(rows) for rows in history.values()),
        'context_fields': list(CONTEXT_FIELDS),
        'source_history_rows': len(bundle['history']),
        'weights_unchanged': True,
        'source_history_preserved': True,
    }
    return result


def pack_model(model_path, source_path, destination):
    import joblib
    model_path, source_path, destination = map(Path, (model_path, source_path, destination))
    if destination.exists():
        raise ValueError('Output already exists; choose a new deployment-artifact path.')
    spec = importlib.util.spec_from_file_location('courtiq_frozen_packing_model', source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bundle = joblib.load(model_path)
    packed = compact_bundle(bundle, module)
    packed['deployment_compaction']['source_artifact_sha256'] = file_sha256(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects an existing artifact if another packer wins.
    created = False
    try:
        with destination.open('xb') as handle:
            created = True
            joblib.dump(packed, handle, compress=3)
    except Exception:
        if created:
            destination.unlink(missing_ok=True)
        raise
    return dict(packed['deployment_compaction'], artifact_sha256=file_sha256(destination),
                artifact_bytes=destination.stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/courtiq/model.joblib')
    parser.add_argument('--source', default='courtiq/model.py')
    parser.add_argument('--out', default='models/courtiq/deployment.joblib')
    args = parser.parse_args()
    try:
        report = pack_model(args.model, args.source, args.out)
    except (ValueError, OSError) as error:
        parser.exit(1, str(error) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
