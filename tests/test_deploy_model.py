"""Serving-artifact compaction retains every query-relevant observation."""
import math
from pathlib import Path
import tempfile
import unittest

from courtiq import model
from courtiq.deploy_model import CONTEXT_FIELDS, compact_bundle, file_sha256, pack_model


def fixture():
    snapshot = model.pd.Timestamp('2026-06-16')
    engine = model.TennisFeatureEngine()
    now = model._sf_stamp(snapshot)
    sipko_edge = math.floor(now / engine.sipko.refresh) * engine.sipko.refresh - engine.sipko.history_days * 86400
    momentum_edge = now - engine.momentum.window * 86400
    dates = [model.pd.Timestamp('2000-01-01')]
    for edge in [sipko_edge, momentum_edge]:
        dates += [model.pd.Timestamp(edge + offset * 86400, unit='s') for offset in [-1, 0, 1]]
    dates += [model.pd.Timestamp('2026-05-25')]
    rows = []
    for i, date in enumerate(sorted(dates)):
        winner, loser = str(i % 3 + 1), str((i + 1) % 3 + 1)
        row = dict(date=date, known_at=date + model.pd.Timedelta(days=21),
                   winner_id=winner, loser_id=loser, winner_name='Player ' + winner,
                   loser_name='Player ' + loser, completed=True, surface=('Hard','Clay','Grass')[i % 3],
                   match_id='match-' + str(i), event_key='example', retirement_proxy=False,
                   minutes=100, score='6-4 6-4')
        for prefix in ['w_', 'l_']:
            for key, value in dict(svpt=70, **{'1stIn':40,'1stWon':30,'2ndWon':15},
                                   ace=5, df=2, bpSaved=2, bpFaced=4).items():
                row[prefix + key] = value
        engine.advance(date)
        engine.schedule(row)
        rows.append(row)
    engine.advance(snapshot)
    return dict(source_sha256=file_sha256(model.__file__), engine_state=engine.export_state(),
                history=model.pd.DataFrame(rows), estimator={'sentinel':'fitted weights'},
                config=engine.config)


def assert_equal_features(test, expected, actual):
    test.assertEqual(expected.keys(), actual.keys())
    for key in expected:
        x, y = expected[key], actual[key]
        if math.isnan(x):
            test.assertTrue(math.isnan(y), key)
        else:
            test.assertEqual(x, y, key)


class DeploymentModelTests(unittest.TestCase):
    def test_feature_math_is_unchanged_across_surfaces_and_future_dates(self):
        source = fixture()
        packed = compact_bundle(source, model)
        original_engine = model.TennisFeatureEngine.from_state(source['config'], model.ChartedShotStore(), source['engine_state'])
        packed_engine = model.TennisFeatureEngine.from_state(packed['config'], model.ChartedShotStore(), packed['engine_state'])
        for day in ['2026-06-16', '2026-07-30', '2027-06-15']:
            for surface in ['Hard','Clay','Grass','Carpet']:
                kwargs = dict(surface=surface, asof=model.tennis_date(day), event='example',
                              level='M', best_of=5, round_name='F')
                assert_equal_features(self, original_engine.matchup('1','2',**kwargs),
                                      packed_engine.matchup('1','2',**kwargs))
                self.assertEqual(original_engine.sipko.snapshot('1',surface,day),
                                 packed_engine.sipko.snapshot('1',surface,day))

    def test_preserves_all_source_history_context_rows_and_weights(self):
        source = fixture()
        packed = compact_bundle(source, model)
        self.assertIs(packed['history'], source['history'])
        self.assertIs(packed['estimator'], source['estimator'])
        self.assertEqual(packed['engine_state']['history'].keys(), source['engine_state']['history'].keys())
        for player, rows in source['engine_state']['history'].items():
            kept = packed['engine_state']['history'][player]
            self.assertEqual(len(rows), len(kept))
            for before, after in zip(rows, kept):
                self.assertEqual(after, {name:before[name] for name in CONTEXT_FIELDS})
        self.assertNotIn('deployment_compaction', source)
        self.assertGreater(len(source['engine_state']['sipko']['seen']),len(packed['engine_state']['sipko']['seen']))

    def test_surface_bucket_boundary_is_retained_inclusively(self):
        source = fixture()
        packed = compact_bundle(source, model)
        cutoff = packed['deployment_compaction']['sipko_cutoff_timestamp']
        dates = [r['date'] for rows in packed['engine_state']['sipko']['records'].values() for r in rows]
        self.assertIn(cutoff, dates)
        self.assertTrue(all(stamp >= cutoff for stamp in dates))
        for player, rows in packed['engine_state']['sipko']['records'].items():
            self.assertEqual(packed['engine_state']['sipko']['times'][player], [r['known'] for r in rows])

    def test_rejects_unreleased_snapshot_and_misaligned_indices(self):
        source = fixture()
        source['engine_state']['pending'] = [('unreleased',)]
        with self.assertRaisesRegex(ValueError,'released'):
            compact_bundle(source, model)
        source = fixture()
        first = next(iter(source['engine_state']['sipko']['times']))
        source['engine_state']['sipko']['times'][first].pop()
        with self.assertRaisesRegex(ValueError,'misaligned'):
            compact_bundle(source, model)

    def test_rejects_wrong_source_and_repeat_compaction(self):
        source = fixture()
        source['source_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError,'exact'):
            compact_bundle(source, model)
        packed = compact_bundle(fixture(), model)
        with self.assertRaisesRegex(ValueError,'already'):
            compact_bundle(packed, model)

    def test_export_does_not_overwrite_existing_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'existing.joblib'
            target.write_bytes(b'keep')
            with self.assertRaisesRegex(ValueError,'already exists'):
                pack_model('unused.joblib', model.__file__, target)
            self.assertEqual(target.read_bytes(), b'keep')


if __name__ == '__main__':
    unittest.main()
