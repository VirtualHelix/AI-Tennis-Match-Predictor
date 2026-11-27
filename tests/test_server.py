"""Focused CourtIQ HTTP integration checks using a trained synthetic fixture.

The fixture verifies software behavior, not tennis forecasting accuracy. These
tests run the real single-file training CLI, load its saved model, and query a
real loopback ThreadingHTTPServer. No fixture is included in the website UI.
"""
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import date, datetime, timedelta, timezone
import http.client
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from courtiq.server import CourtIQService, ThreadingHTTPServer, make_handler


SOURCE = Path(__file__).resolve().parents[1] / 'courtiq' / 'model.py'


def synthetic_atp():
    """Small deterministic format fixture; players and results are synthetic."""
    rng = np.random.default_rng(106)
    rows = []
    for year in range(2019, 2027):
        for week in range(18):
            day = pd.Timestamp(year=year, month=1, day=4) + pd.Timedelta(days=7 * week)
            players = rng.permutation(8)
            for match in range(4):
                first, second = players[2 * match:2 * match + 2]
                w, l = (first, second) if rng.random() < .5 + .04 * (second - first) else (second, first)
                row = dict(tourney_date=int(day.strftime('%Y%m%d')), tourney_id='%s-%s' % (year, week),
                           tourney_name='Event %s' % (week % 4), match_num=match,
                           winner_id=int(1000 + w), loser_id=int(1000 + l),
                           winner_name='Player %s' % w, loser_name='Player %s' % l,
                           winner_rank=int(10 + 4 * w), loser_rank=int(10 + 4 * l),
                           winner_rank_points=int(5000 - 300 * w), loser_rank_points=int(5000 - 300 * l),
                           winner_age=23 + w + year - 2019, loser_age=23 + l + year - 2019,
                           winner_ht=180 + int(w), loser_ht=180 + int(l),
                           surface='Clay' if week % 3 == 0 else 'Hard',
                           tourney_level='G' if week % 6 == 0 else 'A', round='R32',
                           best_of=5 if week % 6 == 0 else 3, minutes=90 + int(w + l), score='6-4 6-4')
                for prefix, won in [('w_', 30), ('l_', 25)]:
                    row.update({prefix + 'svpt':60, prefix + '1stIn':40, prefix + '1stWon':won,
                                prefix + '2ndWon':10, prefix + 'ace':5, prefix + 'df':2,
                                prefix + 'bpSaved':3, prefix + 'bpFaced':5})
                rows.append(row)
    return pd.DataFrame(rows)


def serve(service):
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(service, html='<html>CourtIQ test shell</html>'))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def request(server, method='GET', path='/api/status', payload=None, raw=None, headers=None):
    outgoing = dict(headers or {})
    if payload is not None:
        raw = json.dumps(payload).encode()
        outgoing.setdefault('Content-Type', 'application/json')
    connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=30)
    try:
        connection.request(method, path, body=raw, headers=outgoing)
        response = connection.getresponse()
        body = response.read().decode('utf-8')
        parsed = json.loads(body) if response.getheader('Content-Type', '').startswith('application/json') else body
        return response.status, dict(response.getheaders()), parsed
    finally:
        connection.close()


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        raw = cls.root / 'raw'
        raw.mkdir()
        rows = synthetic_atp()
        for year, part in rows.groupby(rows.tourney_date // 10000):
            part.to_csv(raw / ('atp_matches_%s.csv' % year), index=False)
        cls.model = cls.root / 'model' / 'model.joblib'
        command = [sys.executable, str(SOURCE), 'train', '--data-dir', str(raw),
                   '--updates-dir', str(cls.root / 'no_updates'),
                   '--charting-dir', str(cls.root / 'no_charts'), '--out', str(cls.model.parent),
                   '--training-start', '2019', '--iterations', '3', '--threads', '1']
        completed = subprocess.run(command, cwd=cls.root, text=True, capture_output=True, timeout=180)
        if completed.returncode:
            raise RuntimeError('Synthetic fixture training failed:\n' + completed.stdout + completed.stderr)
        cls.service = CourtIQService(cls.root, model_path=cls.model, source_path=SOURCE)
        if not cls.service.status()['ready']:
            raise RuntimeError('Synthetic fixture could not load: ' + str(cls.service.error))
        cls.server, cls.thread = serve(cls.service)
        cls.forecast = cls.service.status()['default_date']

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.temp.cleanup()

    def setUp(self):
        with self.service.lock:
            self.service.cache.clear()

    def matchup(self, **changes):
        payload = dict(player_a='Player 0', player_b='Player 1', surface='Hard', as_of=self.forecast,
                       event='Event 2', level='A', best_of=3, round='R32')
        payload.update(changes)
        return payload

    def post(self, payload):
        return request(self.server, 'POST', '/api/predict', payload=payload)

    def stream(self, payload, server=None):
        status, headers, body = request(server or self.server, 'POST', '/api/predict/stream', payload=payload)
        self.assertEqual(headers.get('Content-Type'), 'application/x-ndjson; charset=utf-8')
        self.assertIsInstance(body, str)
        self.assertTrue(body.endswith('\n'), 'Every stream event must be newline-delimited.')
        return status, headers, [json.loads(line) for line in body.splitlines() if line]

    def assert_stream_terminal(self, events, terminal_type):
        self.assertTrue(events)
        terminals = [event for event in events if event['type'] in ('result', 'error')]
        self.assertEqual(len(terminals), 1, events)
        self.assertEqual(events[-1], terminals[0])
        self.assertEqual(events[-1]['type'], terminal_type)
        progress = events[:-1]
        self.assertTrue(progress)
        self.assertTrue(all(event['type'] == 'progress' for event in progress))
        values = [event['progress'] for event in progress]
        self.assertTrue(all(isinstance(v, (int, float)) and not isinstance(v, bool)
                            and math.isfinite(v) and 0 <= v <= 100 for v in values))
        self.assertEqual(values, sorted(values))
        for event in progress:
            self.assertIsInstance(event['stage'], str)
            self.assertTrue(event['message'].strip())
        return events[-1]

    def test_results_routes_refresh_without_accepting_browser_credentials(self):
        class Feed:
            called = False
            def status(self):
                return {'configured':True,'state':'ready','revision':1,
                        'progress':{'percent':10,'message':'Fetching recorded results.'},
                        'delay_seconds':30,'data_delay_seconds':900}
            def refresh(self, force=False):
                self.called = force
                return dict(self.status(),state='syncing')
        feed = Feed()
        with patch.object(self.service, 'results_feed', feed):
            code, _, status = request(self.server,path='/api/results/status')
            self.assertEqual(code,200)
            self.assertEqual(status['progress'],10)
            self.assertEqual(status['retry_after_seconds'],30)
            self.assertEqual(status['delay_seconds'],900)
            code, _, status = request(self.server,'POST','/api/results/sync',payload={})
            self.assertEqual(code,200)
            self.assertEqual(status['state'],'syncing')
            self.assertTrue(feed.called)
            code, _, _ = request(self.server,'POST','/api/results/sync',payload={'api_key':'do-not-accept'})
            self.assertEqual(code,400)
            code, _, _ = request(self.server,'POST','/api/results/sync',payload={},
                                  headers={'Origin':'https://unrelated.example'})
            self.assertEqual(code,403)

    def test_feed_revision_updates_history_without_changing_model_probability(self):
        now = datetime.now(timezone.utc)
        played = now.date()-timedelta(days=1)
        class Feed:
            # Replacing the feed fixture is itself a new data revision.
            revision = 100
            first_won = True
            def status(inner):
                return {'configured':True,'state':'ready','revision':inner.revision,
                        'coverage_complete':True,'coverage_year':now.year,
                        'covered_ranges':[{'start':'%s-01-01'%now.year,'end':now.date().isoformat()}],
                        'last_sync':now.isoformat(),'matches':1,'data_delay_seconds':900}
            def snapshot(inner):
                winner,loser = ('Player 0','Player 1') if inner.first_won else ('Player 1','Player 0')
                return {'status':inner.status(),'records':[{
                    'match_id':'sportstats:fixture-1','match_date':played.isoformat(),
                    'winner':{'provider_id':'external-'+winner[-1],'name':winner},
                    'loser':{'provider_id':'external-'+loser[-1],'name':loser},
                    'event':'Fixture API event','surface':'Hard','score':'6-4 6-4','score_kind':'games',
                    'source':'sportstats','date_basis':'match_date','observed_at':now.isoformat()}]}
        feed = Feed()
        payload = self.matchup(event='API history revision')
        saved = self.post(payload)[2]
        with patch.object(self.service,'results_feed',feed):
            first = self.post(payload)[2]
            self.assertEqual(first['prediction'],saved['prediction'])
            self.assertEqual(first['math'],saved['math'])
            self.assertEqual(first['comparison']['data_through'],saved['comparison']['data_through'])
            self.assertEqual(first['recent_matches']['players'][0]['matches'][0]['match_id'],
                             'sportstats:fixture-1')
            self.assertEqual(first['head_to_head']['ytd']['wins_a'],1)
            feed.first_won=False
            feed.revision=101
            second = self.post(payload)[2]
            self.assertFalse(second['cached'])
            self.assertEqual(second['prediction'],saved['prediction'])
            self.assertEqual(second['head_to_head']['ytd']['wins_a'],0)
            self.assertEqual(second['head_to_head']['ytd']['wins_b'],1)
            self.assertTrue(self.post(payload)[2]['cached'])

    def test_stream_matches_standard_response_with_one_terminal_result(self):
        payload = self.matchup(surface='Clay', level='G', best_of=5, event='Stream equivalence')
        expected = self.post(payload)[2]
        with self.service.lock:
            self.service.cache.clear()
        status, headers, events = self.stream(payload)
        self.assertEqual(status, 200)
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertEqual(headers['Connection'], 'close')
        terminal = self.assert_stream_terminal(events, 'result')
        self.assertEqual(terminal['data'], expected)
        self.assertFalse(terminal['data']['cached'])
        self.assertEqual([(e['stage'], e['progress']) for e in events[:-1]],
                         [('queued', 4), ('validating', 8), ('validated', 12),
                          ('predicting', 18), ('comparison', 62), ('explanation', 84), ('complete', 100)])

    def test_stream_flushes_progress_while_model_is_still_running(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        original = self.service.predictor.predict

        def paused_model(*args, **kwargs):
            entered.set()
            try:
                if not release.wait(5):
                    raise RuntimeError('Test did not release the inference gate.')
                return original(*args, **kwargs)
            finally:
                finished.set()

        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=10)
        try:
            with patch.object(self.service.predictor, 'predict', side_effect=paused_model):
                connection.request('POST', '/api/predict/stream',
                                   body=json.dumps(self.matchup(event='Flush before inference')).encode(),
                                   headers={'Content-Type':'application/json'})
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                first = json.loads(response.readline())
                self.assertEqual(first['type'], 'progress')
                self.assertEqual(first['stage'], 'queued')
                self.assertTrue(entered.wait(5))
                self.assertFalse(finished.is_set(), 'Progress must arrive before inference completes.')
                release.set()
                events = [first] + [json.loads(line) for line in response.read().splitlines() if line]
                self.assert_stream_terminal(events, 'result')
        finally:
            release.set()
            connection.close()

    def test_stream_cache_reports_reuse_without_rerunning_model_or_math(self):
        payload = self.matchup(event='Cached stream')
        expected = self.post(payload)[2]
        with patch.object(self.service.predictor, 'predict', side_effect=AssertionError('Unexpected inference')) as model, \
             patch.object(self.service, '_comparison', side_effect=AssertionError('Unexpected comparison')) as comparison, \
             patch.object(self.service, '_prediction_math', side_effect=AssertionError('Unexpected explanation')) as explanation:
            status, _, events = self.stream(payload)
            self.assertEqual(status, 200)
            terminal = self.assert_stream_terminal(events, 'result')
            self.assertTrue(terminal['data']['cached'])
            self.assertEqual(terminal['data'], dict(expected, cached=True))
            self.assertEqual([(e['stage'], e['progress']) for e in events[:-1]],
                             [('queued', 4), ('validating', 8), ('validated', 12), ('cache_hit', 95), ('complete', 100)])
            model.assert_not_called()
            comparison.assert_not_called()
            explanation.assert_not_called()

    def test_stream_semantic_errors_stop_before_model_work(self):
        invalid = [self.matchup(player_b='Player 0'), self.matchup(player_b='No Such Player'),
                   self.matchup(surface='Moon'), self.matchup(best_of=True), []]
        with patch.object(self.service.predictor, 'predict', side_effect=AssertionError('Invalid input reached inference')) as model:
            for payload in invalid:
                with self.subTest(payload=payload):
                    status, _, events = self.stream(payload)
                    self.assertEqual(status, 200)  # Headers precede semantic validation.
                    terminal = self.assert_stream_terminal(events, 'error')
                    self.assertEqual(terminal['code'], 'invalid_request')
                    self.assertIsInstance(terminal['error'], str)
                    self.assertTrue(terminal['error'])
                    self.assertNotIn('data', terminal)
                    self.assertEqual([e['stage'] for e in events[:-1]], ['queued', 'validating'])
            model.assert_not_called()
        self.assertEqual(len(self.service.cache), 0)

    def test_stream_http_errors_remain_normal_json_before_stream_start(self):
        cases = [(b'{', {'Content-Type':'application/json'}, 400),
                 (b'{}', {'Content-Type':'text/plain'}, 415),
                 (b'x' * 8193, {'Content-Type':'application/json'}, 413),
                 (b'{}', {'Content-Type':'application/json', 'Origin':'https://example.com'}, 403)]
        for raw, headers, expected in cases:
            with self.subTest(expected=expected):
                status, received, body = request(self.server, 'POST', '/api/predict/stream', raw=raw, headers=headers)
                self.assertEqual(status, expected)
                self.assertEqual(received['Content-Type'], 'application/json; charset=utf-8')
                self.assertIn('error', body)
                self.assertNotIn('prediction', body)

    def test_stream_model_failure_emits_one_error_and_no_complete_or_result(self):
        for failure, expected_code in [(ValueError('Fixture model validation failed'), 'invalid_request'),
                                       (RuntimeError('Fixture internal detail'), 'server_error')]:
            with self.subTest(expected_code=expected_code), \
                 patch.object(self.service.predictor, 'predict', side_effect=failure):
                status, _, events = self.stream(self.matchup(event='Failure stream'))
                self.assertEqual(status, 200)
                terminal = self.assert_stream_terminal(events, 'error')
                self.assertEqual(terminal['code'], expected_code)
                self.assertNotIn('complete', [e['stage'] for e in events[:-1]])
                self.assertEqual(events[-2]['stage'], 'predicting')
                self.assertEqual(len(self.service.cache), 0)
                if expected_code == 'server_error':
                    self.assertNotIn('Fixture internal detail', terminal['error'])

    def test_stream_missing_model_has_no_probabilities(self):
        service = CourtIQService(self.root, model_path=self.root / 'missing_stream.joblib', source_path=SOURCE)
        server, thread = serve(service)
        try:
            status, _, events = self.stream(self.matchup(), server=server)
            self.assertEqual(status, 200)
            terminal = self.assert_stream_terminal(events, 'error')
            self.assertEqual(terminal['code'], 'model_unavailable')
            self.assertEqual([e['stage'] for e in events[:-1]], ['queued', 'validating'])
            self.assertNotIn('data', terminal)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_disconnected_stream_releases_engine_for_the_next_request(self):
        service = CourtIQService(self.root, model_path=self.model, source_path=SOURCE)
        server, thread = serve(service)
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        original = service.predictor.predict

        def paused_model(*args, **kwargs):
            entered.set()
            try:
                if not release.wait(5):
                    raise RuntimeError('Test did not release the disconnected inference gate.')
                return original(*args, **kwargs)
            finally:
                finished.set()

        connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=10)
        response = None
        try:
            with patch.object(service.predictor, 'predict', side_effect=paused_model):
                connection.request('POST', '/api/predict/stream',
                                   body=json.dumps(self.matchup(event='Disconnected stream')).encode(),
                                   headers={'Content-Type':'application/json'})
                response = connection.getresponse()
                self.assertEqual(json.loads(response.readline())['type'], 'progress')
                self.assertTrue(entered.wait(5))
                response.close()
                connection.close()
                release.set()
                self.assertTrue(finished.wait(5))
                self.assertTrue(request(server)[2]['ready'])
                status, _, answer = request(server, 'POST', '/api/predict',
                                             payload=self.matchup(event='After disconnected stream'))
                self.assertEqual(status, 200, answer)
                self.assertAlmostEqual(answer['prediction']['probability_a'] + answer['prediction']['probability_b'], 1.)
                self.assertAlmostEqual(answer['math']['probability_a'], answer['prediction']['probability_a'])
        finally:
            release.set()
            if response is not None:
                response.close()
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_real_training_load_status_players_and_evaluation(self):
        status, _, data = request(self.server)
        self.assertEqual(status, 200)
        self.assertTrue(data['ready'])
        self.assertEqual(data['model']['trained_before'], '2025-01-01')
        self.assertFalse(data['model']['charting_enabled'])
        self.assertGreaterEqual(data['default_date'], data['min_date'])
        self.assertEqual(data['notice'], 'Historical model snapshot')
        status, _, players = request(self.server, path='/api/players')
        self.assertEqual(status, 200)
        self.assertEqual(len(players['players']), 8)
        self.assertTrue(all(p['matches'] >= 10 for p in players['players']))
        status, _, evaluation = request(self.server, path='/api/evaluation')
        self.assertEqual(status, 200)
        self.assertEqual(evaluation['source'], 'saved_model')
        self.assertEqual(evaluation['report']['validation_year'], 2025)
        self.assertEqual(sum(b['count'] for b in evaluation['calibration']), evaluation['report']['model']['matches'])

    def test_prediction_context_probabilities_and_historical_statistics(self):
        payload = self.matchup(surface='Clay', level='G', best_of=5, round='SF')
        status, headers, data = self.post(payload)
        self.assertEqual(status, 200, data)
        self.assertEqual(headers['Content-Type'], 'application/json; charset=utf-8')
        self.assertNotIn('Access-Control-Allow-Origin', headers)
        p = data['prediction']
        self.assertTrue(0 <= p['probability_a'] <= 1)
        self.assertAlmostEqual(p['probability_a'] + p['probability_b'], 1., places=13)
        self.assertEqual((p['surface'], p['event_level'], p['best_of'], p['round']), ('Clay', 'G', 5, 'SF'))
        self.assertEqual(p['forecast_date'], payload['as_of'])
        self.assertEqual(p['event'], payload['event'])
        self.assertEqual(data['comparison']['data_through'], p['latest_available_tournament_start'])
        self.assertFalse(data['cached'])
        for player in data['comparison']['players']:
            self.assertEqual(player['recent_form']['played'], 10)
            self.assertLess(player['history']['last_known_at'], payload['as_of'])
            self.assertIsNotNone(player['rank']['as_of_tournament_start'])
            self.assertTrue(0 <= player['recent_form']['win_rate'] <= 1)
            for rate in player['rates'].values():
                self.assertGreaterEqual(rate['weighted_opportunities'], 0)
                self.assertTrue(rate['value'] is None or 0 <= rate['value'] <= 1)

    def test_swapping_players_complements_real_model_probability(self):
        _, _, original = self.post(self.matchup())
        status, _, swapped = self.post(self.matchup(player_a='Player 1', player_b='Player 0'))
        self.assertEqual(status, 200, swapped)
        self.assertAlmostEqual(original['prediction']['probability_a'] + swapped['prediction']['probability_a'], 1., places=12)
        self.assertEqual(original['comparison']['players'][0], swapped['comparison']['players'][1])

    def test_missing_round_and_date_use_unspecified_round_and_automatic_date(self):
        payload = self.matchup()
        del payload['round']
        del payload['as_of']
        status, _, answer = self.post(payload)
        self.assertEqual(status, 200, answer)
        self.assertEqual(answer['prediction']['forecast_date'], self.service.status()['default_date'])
        self.assertEqual(answer['prediction']['round'], '')
        self.assertEqual(answer['math']['round_policy'], 'unspecified')
        round_values = {k:v for k,v in answer['math']['feature_values'].items() if k.startswith('context_round_')}
        self.assertTrue(round_values)
        self.assertTrue(all(v == 0 for v in round_values.values()))

    def test_players_and_surface_use_the_standard_match_defaults(self):
        minimal = {'player_a': 'Player 0', 'player_b': 'Player 1', 'surface': 'Grass'}
        status, _, answer = self.post(minimal)
        self.assertEqual(status, 200, answer)
        explicit = dict(minimal, level='A', best_of=3, event='', round='',
                        as_of=self.service.status()['default_date'])
        explicit_status, _, reference = self.post(explicit)
        self.assertEqual(explicit_status, 200, reference)
        self.assertEqual(answer['prediction'], reference['prediction'])
        self.assertEqual(answer['math']['feature_values'], reference['math']['feature_values'])
        prediction = answer['prediction']
        self.assertEqual((prediction['surface'], prediction['event_level'],
                          prediction['best_of'], prediction['event'], prediction['round']),
                         ('Grass', 'A', 3, '', ''))

    def test_methodology_uses_saved_configuration_and_actual_components(self):
        status, _, answer = request(self.server, path='/api/methodology')
        self.assertEqual(status, 200, answer)
        self.assertEqual(answer['source'], 'saved_model')
        bundle = self.service.predictor.bundle
        estimator = bundle['estimator']
        config = answer['config']
        self.assertEqual(config['training_start'], 2019)  # Fixture differs from the reported-run fallback.
        self.assertEqual(config['training_start'], estimator['training_config']['start'])
        self.assertEqual(config['training_half_life_years'], estimator['training_config']['half_life'])
        for key in ('history_days', 'history_discount', 'prior_points'):
            self.assertEqual(config[key], bundle['config'][key])
        self.assertEqual(config['catboost_weight'], estimator['catboost_weight'])
        self.assertEqual(config['logistic_weight'], 1 - estimator['catboost_weight'])
        self.assertEqual(config['features'], len(estimator['features']))
        self.assertEqual(answer['features'], estimator['features'])
        self.assertEqual(sum(group['count'] for group in answer['feature_groups']), len(estimator['features']))
        self.assertEqual(answer['interactions'], [list(pair) for pair in estimator['interaction_pairs']])
        self.assertEqual(answer['prior_rates'], self.service.module._SF_PRIOR_RATES)
        self.assertEqual(answer['models']['logistic_active'], 'logistic' in estimator['models'])
        self.assertEqual(answer['models']['catboost_active'], 'catboost' in estimator['models'])
        if 'catboost' in estimator['models']:
            self.assertEqual(answer['models']['catboost_trees'], 3)
        if 'logistic' in estimator['models']:
            self.assertEqual(answer['models']['logistic_coefficient_count'],
                             estimator['models']['logistic']['classifier'].coef_.shape[1])

    def test_prediction_math_reconstructs_result_and_reverses_components(self):
        status, _, original = self.post(self.matchup(surface='Clay', level='G', best_of=5))
        self.assertEqual(status, 200, original)
        status, _, reverse = self.post(self.matchup(player_a='Player 1', player_b='Player 0', surface='Clay', level='G', best_of=5))
        self.assertEqual(status, 200, reverse)
        detail, swapped = original['math'], reverse['math']
        components = detail['components']
        self.assertEqual(set(components), set(self.service.predictor.bundle['estimator']['models']))
        self.assertAlmostEqual(sum(c['weight'] for c in components.values()), 1.)
        total = 0.
        for name, component in components.items():
            self.assertTrue(0 <= component['forward'] <= 1)
            self.assertTrue(0 <= component['reverse'] <= 1)
            symmetric = (component['forward'] + 1 - component['reverse']) / 2
            self.assertAlmostEqual(component['symmetric'], symmetric, places=13)
            self.assertAlmostEqual(component['forward'], swapped['components'][name]['reverse'], places=13)
            self.assertAlmostEqual(component['reverse'], swapped['components'][name]['forward'], places=13)
            self.assertAlmostEqual(component['symmetric'] + swapped['components'][name]['symmetric'], 1., places=13)
            total += component['weight'] * symmetric
        self.assertAlmostEqual(total, original['prediction']['probability_a'], places=12)
        self.assertAlmostEqual(total, detail['probability_a'], places=13)
        self.assertAlmostEqual(detail['probability_b'], 1 - total, places=13)
        self.assertAlmostEqual(detail['probability_a'] + swapped['probability_a'], 1., places=12)
        for name, value in detail['feature_values'].items():
            reversed_value = swapped['feature_values'][name]
            if value is None:
                self.assertIsNone(reversed_value)
            else:
                self.assertAlmostEqual(reversed_value, -value if name.startswith('diff_') else value, places=12)
        elo = detail['elo']
        self.assertAlmostEqual(elo['probability_a'], 1 / (1 + 10 ** ((elo['b'] - elo['a']) / 400)), places=13)
        self.assertAlmostEqual(elo['surface_probability_a'], 1 / (1 + 10 ** ((elo['surface_b'] - elo['surface_a']) / 400)), places=13)

    def test_head_to_head_uses_full_saved_history_including_older_meetings(self):
        payload = self.matchup(surface='Clay')
        status, _, answer = self.post(payload)
        self.assertEqual(status, 200, answer)
        h2h = answer['head_to_head']
        self.assertEqual(h2h, answer['comparison']['head_to_head'])
        saved = self.service.predictor.bundle['history']
        asof = pd.Timestamp(payload['as_of'])
        pair = ((saved.winner_id == '1000') & (saved.loser_id == '1001')) | ((saved.winner_id == '1001') & (saved.loser_id == '1000'))
        eligible = saved[pair & saved.completed & (saved.known_at < asof)].drop_duplicates('match_id')
        old = eligible[eligible.date < asof - pd.Timedelta(days=self.service.predictor.bundle['config']['history_days'])]
        self.assertGreater(len(old), 0, 'Fixture must distinguish full history from the feature window.')
        for field, rows in [('overall', eligible), ('ytd', eligible[eligible.date.dt.year == asof.year]),
                            ('surface', eligible[eligible.surface == 'Clay'])]:
            self.assertEqual(h2h[field]['played'], len(rows))
            self.assertEqual(h2h[field]['wins_a'], int((rows.winner_id == '1000').sum()))
            self.assertEqual(h2h[field]['wins_b'], int((rows.winner_id == '1001').sum()))
        self.assertEqual(h2h['history_start'], saved.date.min().date().isoformat())
        self.assertEqual(h2h['year'], asof.year)

    def test_head_to_head_deduplicates_and_excludes_unreleased_incomplete_rows(self):
        service = CourtIQService(self.root, model_path=self.model, source_path=SOURCE)
        asof = pd.Timestamp(self.forecast)
        template = service.predictor.records[0]
        year = asof.year

        def meeting(key, when, winner, surface='Hard', **changes):
            day = pd.Timestamp(when)
            row = dict(template, match_id=key, tourney_id=key, match_num=1, date=day,
                       known_at=day + pd.Timedelta(days=21), winner_id=winner,
                       loser_id='1001' if winner == '1000' else '1000', surface=surface,
                       completed=True, tourney_name='Fixture event', score='6-4 6-4')
            row.update(changes)
            return row

        old_day = asof - pd.Timedelta(days=service.predictor.bundle['config']['history_days'] + 500)
        old = meeting('older_than_feature_window', old_day, '1000')
        prior = meeting('prior_year', '%s-02-01' % (year - 1), '1001', 'Clay')
        this_year = meeting('this_year_hard', '%s-01-02' % year, '1000')
        latest = meeting('this_year_clay', '%s-03-02' % year, '1001', 'Clay', score=float('nan'), tourney_name=None)
        unreleased = meeting('unreleased_at_boundary', '%s-04-01' % year, '1000', known_at=asof)
        incomplete = meeting('retired_or_incomplete', '%s-04-02' % year, '1000', completed=False)
        unrelated = meeting('another_opponent', '%s-04-03' % year, '1000', loser_id='1002')
        service.predictor.records = [old, prior, this_year, latest, copy.deepcopy(old), unreleased, incomplete, unrelated]
        server, thread = serve(service)
        try:
            payload = self.matchup(surface='Hard')
            status, _, result = request(server, path='/api/compare?' + urlencode(payload))
            self.assertEqual(status, 200, result)
            h2h = result['head_to_head']
            self.assertEqual(h2h['overall'], dict(played=4, wins_a=2, wins_b=2))
            self.assertEqual(h2h['ytd'], dict(played=2, wins_a=1, wins_b=1))
            self.assertEqual(h2h['surface'], dict(played=2, wins_a=2, wins_b=0, surface='Hard'))
            self.assertEqual(h2h['latest_meeting']['date'], '%s-03-02' % year)
            self.assertEqual(h2h['latest_meeting']['winner_name'], 'Player 1')
            self.assertIsNone(h2h['latest_meeting']['event'])
            self.assertIsNone(h2h['latest_meeting']['score'])
            payload.update(player_a='Player 1', player_b='Player 0')
            swapped = request(server, path='/api/compare?' + urlencode(payload))[2]['head_to_head']
            self.assertEqual(swapped['surface'], dict(played=2, wins_a=0, wins_b=2, surface='Hard'))
            # Keep the trained state but remove this pair's descriptive history
            # to exercise a genuine empty-meetings branch, not a fabricated 0-0 match.
            with service.lock:
                service.predictor.records = [unrelated, unreleased, incomplete]
            empty = request(server, path='/api/compare?' + urlencode(payload))[2]['head_to_head']
            self.assertEqual(empty['overall'], dict(played=0, wins_a=0, wins_b=0))
            self.assertEqual(empty['ytd'], dict(played=0, wins_a=0, wins_b=0))
            self.assertEqual(empty['surface']['played'], 0)
            self.assertIsNone(empty['latest_meeting'])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_recent_matches_matches_saved_history_and_prediction_contract(self):
        payload = self.matchup(surface='Clay')
        status, _, answer = self.post(payload)
        self.assertEqual(status, 200, answer)
        recent = answer['recent_matches']
        self.assertEqual(recent, answer['comparison']['recent_matches'])
        self.assertEqual(recent, self.service.compare(payload)['recent_matches'])
        self.assertEqual(recent['source'], 'saved_history')
        self.assertEqual(recent['date_basis'], 'tournament_start')
        self.assertEqual(recent['score_basis'], 'winner_first')
        self.assertEqual(recent['as_of'], payload['as_of'])
        self.assertEqual(recent['data_through'], answer['comparison']['data_through'])
        self.assertEqual(self.service.status()['app_version'], '1.5.0')

        saved = self.service.predictor.bundle['history']
        eligible = saved[saved.completed & (saved.known_at < pd.Timestamp(payload['as_of']))]
        eligible = eligible.drop_duplicates('match_id').sort_values(
            ['date', 'tourney_id', 'match_num', 'match_id'], ascending=False)
        pair = ((eligible.winner_id == '1000') & (eligible.loser_id == '1001')) | (
            (eligible.winner_id == '1001') & (eligible.loser_id == '1000'))
        self.assertEqual([r['match_id'] for r in recent['head_to_head']],
                         eligible[pair].head(3).match_id.tolist())
        names = self.service.predictor.bundle['names']
        for player, player_id in zip(recent['players'], ['1000', '1001']):
            expected = eligible[(eligible.winner_id == player_id) | (eligible.loser_id == player_id)].head(3)
            self.assertEqual(player['id'], player_id)
            self.assertEqual(player['name'], names[player_id])
            self.assertEqual(len(player['matches']), 3)
            self.assertEqual([r['match_id'] for r in player['matches']], expected.match_id.tolist())
            for actual, row in zip(player['matches'], expected.to_dict('records')):
                won = row['winner_id'] == player_id
                opponent = row['loser_id'] if won else row['winner_id']
                self.assertEqual(actual['result'], 'W' if won else 'L')
                self.assertEqual(actual['opponent'], {'id':opponent, 'name':names[opponent]})
                self.assertEqual(actual['winner'], {'id':row['winner_id'], 'name':names[row['winner_id']]})
                self.assertEqual(actual['loser'], {'id':row['loser_id'], 'name':names[row['loser_id']]})
                self.assertEqual(actual['score'], row['score'])
                self.assertEqual(actual['set_scores'], [
                    {'winner_games': 6, 'loser_games': 4,
                     'tiebreak_loser_points': None, 'is_match_tiebreak': False},
                    {'winner_games': 6, 'loser_games': 4,
                     'tiebreak_loser_points': None, 'is_match_tiebreak': False}])
                self.assertEqual(actual['tournament_date'], row['date'].date().isoformat())
                self.assertEqual(actual['surface'], row['surface'])
        self.assertTrue(any(m['surface'] != 'Clay' for p in recent['players'] for m in p['matches']),
                        'Recent general matches must not be filtered to the selected court.')

    def test_recent_matches_release_boundary_dedup_order_missing_values_and_player_swap(self):
        service = CourtIQService(self.root, model_path=self.model, source_path=SOURCE)
        asof = pd.Timestamp(self.forecast)
        template = service.predictor.records[0]
        day = asof - pd.Timedelta(days=40)

        def meeting(key, winner='1000', loser='1001', **changes):
            row = dict(template, match_id=key, tourney_id='same_event', match_num=1,
                       date=day, known_at=day + pd.Timedelta(days=21), winner_id=winner,
                       loser_id=loser, surface='Hard', completed=True,
                       tourney_name='Recorded event', score='7-6(5) 6-3')
            row.update(changes)
            return row

        older = meeting('older', date=day - pd.Timedelta(days=7))
        nine = meeting('nine', winner='1001', loser='1000', match_num='9', surface='Grass')
        ten_a = meeting('ten_a', match_num='10', score=float('nan'), tourney_name=None)
        ten_b = meeting('ten_b', winner='1001', loser='1000', match_num='10', surface='Clay')
        other_a = meeting('other_a', loser='1002', date=day + pd.Timedelta(days=1), surface='Grass')
        other_b = meeting('other_b', winner='1003', date=day + pd.Timedelta(days=2), surface='Clay')
        excluded = [meeting('at_release_boundary', known_at=asof),
                    meeting('not_yet_released', known_at=asof + pd.Timedelta(days=1)),
                    meeting('retirement', completed=False, score='6-4 1-0 RET'),
                    meeting('walkover', completed=False, score='W/O'),
                    meeting('unrelated', winner='1004', loser='1005')]
        service.predictor.records = [nine, *excluded, ten_a, other_b, older, ten_b,
                                     copy.deepcopy(ten_b), other_a]
        payload = self.matchup(surface='Hard')
        recent = service.compare(payload)['recent_matches']
        self.assertEqual([r['match_id'] for r in recent['head_to_head']], ['ten_b', 'ten_a', 'nine'])
        self.assertEqual([r['match_id'] for r in recent['players'][0]['matches']],
                         ['other_a', 'ten_b', 'ten_a'])
        self.assertEqual([r['match_id'] for r in recent['players'][1]['matches']],
                         ['other_b', 'ten_b', 'ten_a'])
        self.assertEqual(recent['players'][0]['matches'][1]['result'], 'L')
        self.assertEqual(recent['players'][0]['matches'][1]['score'], '7-6(5) 6-3',
                         'A loss changes the W/L label, never the source score orientation.')
        self.assertEqual(recent['players'][0]['matches'][1]['set_scores'], [
            {'winner_games': 7, 'loser_games': 6,
             'tiebreak_loser_points': 5, 'is_match_tiebreak': False},
            {'winner_games': 6, 'loser_games': 3,
             'tiebreak_loser_points': None, 'is_match_tiebreak': False}])
        self.assertIsNone(recent['head_to_head'][1]['score'])
        self.assertEqual(recent['head_to_head'][1]['set_scores'], [])
        self.assertIsNone(recent['head_to_head'][1]['event'])
        swapped = service.compare(self.matchup(player_a='Player 1', player_b='Player 0'))['recent_matches']
        self.assertEqual(swapped['head_to_head'], recent['head_to_head'])
        self.assertEqual(swapped['players'], list(reversed(recent['players'])))
        # Tied tournament-date records remain stable if source row order changes.
        service.predictor.records.reverse()
        self.assertEqual(service.compare(payload)['recent_matches'], recent)

    def test_recent_matches_do_not_pad_or_limit_to_feature_lookback(self):
        service = CourtIQService(self.root, model_path=self.model, source_path=SOURCE)
        asof = pd.Timestamp(self.forecast)
        old_day = asof - pd.Timedelta(days=service.predictor.bundle['config']['history_days'] + 100)
        template = service.predictor.records[0]
        old = dict(template, match_id='only_old_meeting', date=old_day,
                   known_at=old_day + pd.Timedelta(days=21), winner_id='1000', loser_id='1001',
                   completed=True, score='6-3 6-2')
        unrelated_pair = dict(old, match_id='other_opponent', winner_id='1001', loser_id='1002',
                              date=old_day + pd.Timedelta(days=1))
        service.predictor.records = [old, unrelated_pair]
        recent = service.compare(self.matchup())['recent_matches']
        self.assertEqual([r['match_id'] for r in recent['head_to_head']], ['only_old_meeting'])
        self.assertEqual([len(p['matches']) for p in recent['players']], [1, 2])
        self.assertEqual(recent['players'][0]['matches'][0]['result'], 'W')
        self.assertEqual(recent['players'][1]['matches'][1]['result'], 'L')
        service.predictor.records = [unrelated_pair]
        one = service.compare(self.matchup())['recent_matches']
        self.assertEqual(one['head_to_head'], [])
        self.assertEqual([len(p['matches']) for p in one['players']], [0, 1])
        service.predictor.records = []
        empty = service.compare(self.matchup())['recent_matches']
        self.assertEqual(empty['head_to_head'], [])
        self.assertEqual([p['matches'] for p in empty['players']], [[], []])

    def test_cache_is_context_specific_and_predictions_survive_rewinds(self):
        original = self.post(self.matchup())[2]
        again = self.post(self.matchup())[2]
        self.assertTrue(again['cached'])
        self.assertEqual(original['prediction'], again['prediction'])
        changed = self.post(self.matchup(surface='Clay'))[2]
        self.assertFalse(changed['cached'])
        earlier = self.matchup(as_of=self.service.status()['min_date'], event='Earlier')
        status, _, answer = self.post(earlier)
        self.assertEqual(status, 200, answer)
        fresh = CourtIQService(self.root, model_path=self.model, source_path=SOURCE)
        self.assertEqual(answer['prediction'], fresh.predict(earlier)['prediction'])
        with self.service.lock:
            self.service.cache.clear()
        latest = self.post(self.matchup())[2]
        self.assertEqual(original['prediction'], latest['prediction'])

    def test_comparison_query_preserves_best_of_and_surface(self):
        status, _, data = request(self.server, path='/api/compare?' + urlencode(self.matchup(best_of=5, surface='Clay')))
        self.assertEqual(status, 200, data)
        self.assertEqual(data['surface'], 'Clay')
        self.assertEqual([p['name'] for p in data['players']], ['Player 0', 'Player 1'])

    def test_invalid_inputs_return_controlled_errors(self):
        invalid = [self.matchup(player_b='1000'), self.matchup(player_b='Unknown Player'),
                   self.matchup(best_of=True), self.matchup(best_of=3.0), self.matchup(best_of=4),
                   self.matchup(surface='Moon'), self.matchup(level='Unknown'), self.matchup(round='R3'),
                   self.matchup(as_of='2026-02-30'), self.matchup(as_of='2025-01-01'),
                   self.matchup(as_of=self.forecast + 'T00:00:00Z'),
                   self.matchup(event='Bad\nEvent'), self.matchup(unrecognized='extra'), [], None]
        for payload in invalid:
            with self.subTest(payload=payload):
                if payload is None:
                    status, _, data = request(self.server, 'POST', '/api/predict', raw=b'null', headers={'Content-Type':'application/json'})
                else:
                    status, _, data = self.post(payload)
                self.assertEqual(status, 400, data)
                self.assertIn('error', data)
                self.assertNotIn('prediction', data)
        future = (date.fromisoformat(self.service.status()['max_date']) + timedelta(days=1)).isoformat()
        self.assertEqual(self.post(self.matchup(as_of=future))[0], 400)

    def test_only_trusted_local_origins_and_allowlisted_routes(self):
        for headers in ({'Host':'example.com'}, {'Origin':'https://example.com'}, {'Origin':'null'}):
            with self.subTest(headers=headers):
                self.assertEqual(request(self.server, headers=headers)[0], 403)
        self.assertEqual(request(self.server, headers={'Origin':'http://localhost:%s' % self.server.server_port})[0], 200)
        for path in ('/model/model.joblib', '/server.py', '/../server.py', '/%2e%2e/server.py', '/raw/atp_matches_2025.csv'):
            with self.subTest(path=path):
                self.assertEqual(request(self.server, path=path)[0], 404)
        status, headers, html = request(self.server, path='/')
        self.assertEqual(status, 200)
        self.assertEqual(html, '<html>CourtIQ test shell</html>')
        self.assertTrue(headers['Content-Type'].startswith('text/html'))

    def test_bounded_json_requests_and_repeated_query_fields(self):
        for raw, content_type, expected in ((b'{', 'application/json', 400),
                                           (b'{}', 'text/plain', 415),
                                           (b'x' * 8193, 'application/json', 413)):
            status, _, data = request(self.server, 'POST', '/api/predict', raw=raw, headers={'Content-Type':content_type})
            self.assertEqual(status, expected, data)
        status, _, _ = request(self.server, path='/api/compare?player_a=Player+0&player_a=Player+1')
        self.assertEqual(status, 400)

    def test_concurrent_requests_preserve_each_dates_state(self):
        first = self.service.status()['min_date']
        payloads = [self.matchup(as_of=d, surface=s, event='Concurrency %s' % i)
                    for i, (d, s) in enumerate([(self.forecast, 'Hard'), (first, 'Clay'),
                                               (self.forecast, 'Clay'), (first, 'Hard')])]
        baseline = CourtIQService(self.root, model_path=self.model, source_path=SOURCE)
        expected = [baseline.predict(p) for p in payloads]
        with ThreadPoolExecutor(max_workers=4) as executor:
            received = list(executor.map(self.post, payloads))
        for (status, _, answer), reference in zip(received, expected):
            self.assertEqual(status, 200, answer)
            self.assertEqual(answer['prediction'], reference['prediction'])
            self.assertEqual(answer['comparison'], reference['comparison'])

    def test_missing_model_starts_honestly_without_predictions(self):
        service = CourtIQService(self.root, model_path=self.root / 'missing.joblib', source_path=SOURCE)
        server, thread = serve(service)
        try:
            status, _, data = request(server)
            self.assertEqual(status, 200)
            self.assertFalse(data['ready'])
            self.assertIn('missing', data['error'])
            status, _, data = request(server, 'POST', '/api/predict', payload=self.matchup())
            self.assertEqual(status, 503)
            self.assertEqual(data['error']['code'], 'model_unavailable')
            self.assertNotIn('prediction', data)
            self.assertEqual(request(server, path='/api/players')[0], 503)
            evaluation = request(server, path='/api/evaluation')[2]
            self.assertEqual(evaluation['source'], 'reported_run')
            self.assertEqual(evaluation['calibration'], [])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_mismatched_training_source_cannot_load_as_ready(self):
        copied = self.root / 'changed_training_source.py'
        copied.write_text(SOURCE.read_text() + '\n# Deliberate fixture hash mismatch\n')
        service = CourtIQService(self.root, model_path=self.model, source_path=copied)
        self.assertFalse(service.status()['ready'])
        self.assertIn('exact script version', service.error)


if __name__ == '__main__':
    unittest.main()
