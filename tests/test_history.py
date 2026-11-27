"""Regression checks for identity, availability, overlap and immutable histories."""
import copy
from datetime import date, datetime, timedelta, timezone
import unittest

from courtiq.history import merge_results_history


NOW = datetime(2026, 11, 27, 12, 0, tzinfo=timezone.utc)
NAMES = {'1':'Carlos Alcaraz', '2':'Novak Djokovic', '3':'Alex de Minaur',
         '4':'Alexander Zverev', '5':'Andrey Rublev', '6':'John Smith',
         '7':'James Smith', '8':'Jan Lennard Struff', '9':'Roberto Bautista Agut',
         '10':"Christopher O'Connell"}


def legacy(mid, day, winner='1', loser='2', **changes):
    when = date.fromisoformat(day)
    row = {'match_id':mid, 'date':when, 'known_at':when + timedelta(days=21),
           'winner_id':winner, 'loser_id':loser, 'winner_name':NAMES[winner],
           'loser_name':NAMES[loser], 'surface':'Hard', 'completed':True,
           'score':'6-4 6-3', 'tourney_id':mid, 'match_num':1, 'tourney_name':'Saved event'}
    row.update(changes)
    return row


def api(mid='new', day='2026-10-01', winner='C. Alcaraz', loser='N. Djokovic', **changes):
    row = {'match_id':'sportstats:' + mid, 'match_date':day,
           'winner':{'provider_id':'provider-a', 'name':winner},
           'loser':{'provider_id':'provider-b', 'name':loser},
           'event':'Feed event', 'surface':'Clay', 'score':'6-2 7-6(4)', 'score_kind':'games',
           'source':'sportstats', 'date_basis':'match_date', 'observed_at':'2026-11-27T11:00:00Z'}
    row.update(changes)
    return row


class MergeHistoryTests(unittest.TestCase):
    def setUp(self):
        self.names = copy.deepcopy(NAMES)
        self.legacy = [legacy('old1', '2024-05-01'),
                       legacy('old2', '2025-10-01', '2', '1'),
                       legacy('overlap', '2026-05-25')]
        self.args = {'player_a':'1', 'player_b':'2', 'surface':'Hard', 'as_of':'2026-11-27'}
        record = {'match_id':'overlap', 'tournament_date':'2026-05-25',
                  'winner':{'id':'1', 'name':NAMES['1']}, 'loser':{'id':'2', 'name':NAMES['2']},
                  'score':'6-4 6-3', 'event':'Saved event', 'surface':'Hard'}
        self.old_recent = {'head_to_head':[copy.deepcopy(record)],
                           'players':[{'id':p, 'name':NAMES[p], 'matches':[
                               dict(copy.deepcopy(record), result='W' if p == '1' else 'L',
                                    opponent={'id':'2' if p == '1' else '1', 'name':NAMES['2' if p == '1' else '1']})]}
                                      for p in ('1', '2')],
                           'as_of':'2026-11-27', 'data_through':'2026-05-25', 'source':'saved_history',
                           'date_basis':'tournament_start', 'score_basis':'winner_first', 'scope':'Saved history'}
        self.old_h2h = {'players':[{'id':p, 'name':NAMES[p]} for p in ('1', '2')], 'year':2026,
                        'overall':{'played':3, 'wins_a':2, 'wins_b':1},
                        'ytd':{'played':1, 'wins_a':1, 'wins_b':0},
                        'surface':{'played':3, 'wins_a':2, 'wins_b':1, 'surface':'Hard'},
                        'latest_meeting':{'date':'2026-05-25', 'winner_id':'1', 'winner_name':NAMES['1'],
                                          'event':'Saved event', 'surface':'Hard', 'score':'6-4 6-3'},
                        'history_start':'2024-05-01', 'data_through':'2026-05-25', 'scope':'Saved history'}

    def merge(self, rows, complete=True, **changes):
        status = {'configured':True, 'state':'ready', 'last_sync':'2026-11-27T11:00:00Z',
                  'coverage_complete':complete,
                  'covered_ranges':[{'start':'2026-01-01', 'end':'2026-11-27'}] if complete else []}
        status.update(changes.pop('status', {}))
        inputs = dict(legacy_records=self.legacy, names=self.names, args=self.args,
                      model_data_through='2026-05-25', old_recent=self.old_recent,
                      old_h2h=self.old_h2h, feed_snapshot={'records':rows, 'status':status}, now=NOW)
        inputs.update(changes)
        return merge_results_history(**inputs)

    def test_full_season_replaces_legacy_season_and_keeps_prior_years(self):
        # Different providers describe the same May meeting with different IDs
        # and a match date instead of its tournament start. Replacing the season
        # avoids counting that meeting twice without fuzzy cross-source joins.
        duplicate = api('same-real-match', '2026-05-30', surface='Hard', score='6-4 6-3')
        newer = api('new', '2026-10-01', winner='N. Djokovic', loser='C. Alcaraz')
        recent, h2h = self.merge([duplicate, newer])
        self.assertEqual(h2h['overall'], {'played':4, 'wins_a':2, 'wins_b':2})
        self.assertEqual(h2h['ytd'], {'played':2, 'wins_a':1, 'wins_b':1})
        self.assertEqual(h2h['surface'], {'played':3, 'wins_a':2, 'wins_b':1, 'surface':'Hard'})
        self.assertEqual([r['match_id'] for r in recent['head_to_head']],
                         ['sportstats:new', 'sportstats:same-real-match', 'old2'])
        self.assertEqual(h2h['latest_meeting']['date'], '2026-10-01')
        self.assertEqual(h2h['latest_meeting']['winner_id'], '2')
        self.assertEqual(h2h['latest_meeting']['date_basis'], 'match_date')
        self.assertEqual(h2h['history_start'], '2024-05-01')
        self.assertEqual(recent['model_data_through'], '2026-05-25')
        self.assertEqual(recent['data_through'], '2026-10-01')
        self.assertEqual(recent['date_basis'], 'mixed')
        self.assertIsNone(recent['head_to_head'][0]['tournament_date'])
        self.assertEqual(recent['head_to_head'][0]['match_date'], '2026-10-01')
        self.assertIn('competition coverage can differ', recent['coverage_note'])
        self.assertIn('not official all-time', h2h['coverage_note'])

    def test_partial_backfill_only_overlays_after_legacy_plus_21_and_freezes_totals(self):
        rows = [api('overlap', '2026-05-30'), api('boundary', '2026-06-15'), api('safe', '2026-06-16')]
        recent, h2h = self.merge(rows, complete=False)
        self.assertEqual([r['match_id'] for r in recent['head_to_head']], ['sportstats:safe', 'overlap', 'old2'])
        for field in ('overall', 'ytd', 'surface', 'latest_meeting', 'data_through'):
            self.assertEqual(h2h[field], self.old_h2h[field])
        self.assertEqual(h2h['totals_source'], 'saved_history')
        self.assertEqual(recent['data_through'], '2026-06-16')
        self.assertIn('backfill is incomplete', h2h['coverage_note'])
        self.assertIn('2026-06-15', recent['coverage_note'])

    def test_partial_without_safe_new_records_preserves_existing_cards(self):
        recent, h2h = self.merge([api(day='2026-06-15')], complete=False)
        for key, value in self.old_recent.items():
            self.assertEqual(recent[key], value)
        for key, value in self.old_h2h.items():
            self.assertEqual(h2h[key], value)

    def test_full_coverage_requires_contiguous_ranges_and_explicit_flag(self):
        missing_day = [{'start':'2026-01-01', 'end':'2026-06-14'},
                       {'start':'2026-06-16', 'end':'2026-11-27'}]
        _, h2h = self.merge([api()], status={'covered_ranges':missing_day})
        self.assertEqual(h2h['overall'], self.old_h2h['overall'])
        adjacent = [{'start':'2026-06-15', 'end':'2026-11-27'},
                    {'start':'2026-01-01', 'end':'2026-06-14'}]
        _, h2h = self.merge([api()], status={'covered_ranges':adjacent})
        self.assertEqual(h2h['ytd'], {'played':1, 'wins_a':1, 'wins_b':0})
        self.assertEqual(h2h['overall']['played'], 3)
        self.assertEqual(h2h['totals_source'], 'saved_history+sportstats')
        _, h2h = self.merge([api()], complete=False, status={'covered_ranges':adjacent})
        self.assertEqual(h2h['totals_source'], 'saved_history')

    def test_accents_punctuation_full_surnames_and_multiple_initials(self):
        cases = [('cÁrlos--ALCARAZ', '1'), ('C. Alcaraz', '1'), ('Á. de Minaur', '3'),
                 ('J.L. Struff', '8'), ('R. Bautista Agut', '9'), ('C. Oconnell', '10')]
        for provider_name, player in cases:
            with self.subTest(name=provider_name):
                args = dict(self.args, player_a=player)
                recent, _ = self.merge([api(winner=provider_name)], args=args)
                self.assertEqual(recent['players'][0]['matches'][0]['winner']['id'], player)
                self.assertEqual(recent['players'][0]['matches'][0]['winner']['name'], NAMES[player])

    def test_ambiguous_initials_and_different_full_first_names_remain_unmatched(self):
        cases = ['J. Smith', 'Carlo Alcaraz', 'Alcaraz', 'R. Agut', 'Jan Struff']
        for name in cases:
            with self.subTest(name=name):
                row = api(winner=name)
                recent, h2h = self.merge([row])
                match = recent['players'][1]['matches'][0]
                self.assertEqual(match['winner']['id'], 'sportstats-player:provider-a')
                self.assertEqual(match['winner']['name'], name)
                self.assertEqual(match['opponent']['id'], 'sportstats-player:provider-a')
                self.assertEqual(h2h['ytd']['played'], 0)

    def test_provider_id_never_implicitly_matches_a_local_id(self):
        row = api(winner={'provider_id':'1', 'name':'Unknown Opponent'})
        # api() accepts a string shorthand, so supply the normalized participant.
        row['winner'] = {'provider_id':'1', 'name':'Unknown Opponent'}
        recent, h2h = self.merge([row])
        self.assertEqual(recent['players'][1]['matches'][0]['winner']['id'], 'sportstats-player:1')
        self.assertEqual(h2h['ytd']['played'], 0)

    def test_explicit_alias_is_validated_and_resolves_ambiguous_name(self):
        row = api(winner='J. Smith')
        args = dict(self.args, player_a='6')
        recent, h2h = self.merge([row], args=args, aliases={'provider-a':'6'})
        self.assertEqual(recent['players'][0]['matches'][0]['winner']['name'], 'John Smith')
        self.assertEqual(h2h['ytd'], {'played':1, 'wins_a':1, 'wins_b':0})
        recent, _ = self.merge([api()], aliases={'provider-a':'not-a-local-player'})
        self.assertEqual(recent['players'][1]['matches'][0]['winner']['id'], 'sportstats-player:provider-a')

    def test_unknown_opponent_is_retained_when_one_participant_maps(self):
        row = api(loser='New Person')
        row['loser']['provider_id'] = 'new-person'
        recent, h2h = self.merge([row])
        match = recent['players'][0]['matches'][0]
        self.assertEqual(match['result'], 'W')
        self.assertEqual(match['opponent']['id'], 'sportstats-player:new-person')
        self.assertEqual(match['opponent']['name'], 'New Person')
        self.assertEqual(h2h['ytd']['played'], 0)

    def test_duplicate_ids_use_latest_observation_and_conflicting_ties_are_dropped(self):
        original = api()
        corrected = api(winner='N. Djokovic', loser='C. Alcaraz', observed_at='2026-11-27T11:30:00Z')
        recent, h2h = self.merge([original, copy.deepcopy(original), corrected, copy.deepcopy(corrected)])
        self.assertEqual(h2h['ytd'], {'played':1, 'wins_a':0, 'wins_b':1})
        self.assertEqual(recent['head_to_head'][0]['winner']['id'], '2')
        self.assertEqual(self.merge([corrected, original]), (recent, h2h))
        contradictory = api(winner='N. Djokovic', loser='C. Alcaraz')
        recent, h2h = self.merge([original, contradictory])
        self.assertEqual(recent['head_to_head'], self.old_recent['head_to_head'])
        self.assertEqual(h2h['overall'], self.old_h2h['overall'])

    def test_historical_asof_keeps_original_view_without_later_observations(self):
        args = dict(self.args, as_of='2026-11-26')
        self.assertEqual(self.merge([api()], args=args), (self.old_recent, self.old_h2h))

    def test_future_dates_observations_and_corrupt_rows_do_not_enter_history(self):
        invalid = [api('future-date', day='2026-11-28'),
                   api('future-observation', observed_at='2026-11-27T12:00:01Z'),
                   api('naive-observation', observed_at='2026-11-27T11:00:00'),
                   api('invalid-date', day='2026-02-30'),
                   api('observed-before-match', observed_at='2026-09-30T23:59:59Z'),
                   api('wrong-source', source='somewhere'), api('missing-score', score=None),
                   api('retired', score='6-4 RET'), api('wrong-kind', score_kind='points'),
                   api('old-season', day='2025-11-27'), None, {}]
        bad_player = api('bad-player')
        bad_player['winner'] = {'provider_id':'bad id', 'name':'C. Alcaraz'}
        invalid.append(bad_player)
        for row in invalid:
            with self.subTest(row=row):
                recent, h2h = self.merge([row])
                for field, value in self.old_recent.items():
                    self.assertEqual(recent[field], value)
                for field, value in self.old_h2h.items():
                    self.assertEqual(h2h[field], value)
        recent, h2h = self.merge(invalid + [api('valid')])
        self.assertEqual(h2h['ytd']['played'], 1)
        self.assertEqual(recent['head_to_head'][0]['match_id'], 'sportstats:valid')
        # A future forecast date still cannot reveal a future completed result.
        _, h2h = self.merge([api('valid'), api('future', day='2026-12-01')],
                            args=dict(self.args, as_of='2027-01-01'))
        self.assertEqual(h2h['overall']['played'], 3)
        self.assertEqual(h2h['ytd']['played'], 0)

    def test_incomplete_unreleased_legacy_rows_stay_out_after_rebuild(self):
        extra = [legacy('unreleased', '2026-01-01', known_at=NOW.date()),
                 legacy('retired', '2025-01-01', completed=False),
                 copy.deepcopy(self.legacy[0])]
        _, h2h = self.merge([api()], legacy_records=self.legacy + extra)
        self.assertEqual(h2h['overall'], {'played':3, 'wins_a':2, 'wins_b':1})

    def test_swap_reverses_counts_and_player_lists_not_scores(self):
        recent, h2h = self.merge([api()])
        swapped_recent, swapped_h2h = self.merge([api()], args=dict(self.args, player_a='2', player_b='1'))
        self.assertEqual(swapped_recent['head_to_head'], recent['head_to_head'])
        self.assertEqual(swapped_recent['players'], list(reversed(recent['players'])))
        for field in ('overall', 'ytd', 'surface'):
            self.assertEqual(swapped_h2h[field]['wins_a'], h2h[field]['wins_b'])
            self.assertEqual(swapped_h2h[field]['wins_b'], h2h[field]['wins_a'])
        self.assertEqual(swapped_recent['players'][0]['matches'][0]['result'], 'L')
        self.assertEqual(swapped_recent['players'][0]['matches'][0]['score'], '6-2 7-6(4)')

    def test_sets_only_score_is_preserved_without_inventing_game_scores(self):
        unverified = [{'winner_games':6, 'loser_games':0,
                       'tiebreak_loser_points':None, 'is_match_tiebreak':False}] * 2
        recent, h2h = self.merge([api(score='2-1 sets', score_kind='sets', set_scores=unverified)])
        self.assertEqual(recent['head_to_head'][0]['score'], '2-1 sets')
        self.assertEqual(recent['head_to_head'][0]['score_kind'], 'sets')
        self.assertEqual(recent['head_to_head'][0]['set_scores'], [])
        for player in recent['players']:
            self.assertEqual(player['matches'][0]['set_scores'], [])
        self.assertEqual(h2h['latest_meeting']['score_kind'], 'sets')
        self.assertEqual(h2h['latest_meeting']['set_scores'], [])

    def test_complete_game_scores_reach_all_views_and_preserve_loser_perspective(self):
        score = '2-6 7-6(5) 6-3'
        expected = [
            {'winner_games':2, 'loser_games':6, 'tiebreak_loser_points':None, 'is_match_tiebreak':False},
            {'winner_games':7, 'loser_games':6, 'tiebreak_loser_points':5, 'is_match_tiebreak':False},
            {'winner_games':6, 'loser_games':3, 'tiebreak_loser_points':None, 'is_match_tiebreak':False},
        ]
        # Deliberately contradictory structured cells cannot override the
        # validated winner-first game score in a normalized feed record.
        row = api(score=score, set_scores=[dict(expected[0], winner_games=6)])
        original = copy.deepcopy(row)
        recent, h2h = self.merge([row])
        views = [recent['head_to_head'][0], h2h['latest_meeting']]
        views.extend(player['matches'][0] for player in recent['players'])
        for view in views:
            self.assertEqual(view['score'], score)
            self.assertEqual(view['set_scores'], expected)
        loser_view = recent['players'][1]['matches'][0]
        self.assertEqual(loser_view['result'], 'L')
        self.assertEqual(loser_view['winner']['id'], '1')
        self.assertEqual(loser_view['opponent']['id'], '1')
        self.assertEqual(row, original)

        swapped, _ = self.merge([row], args=dict(self.args, player_a='2', player_b='1'))
        self.assertEqual(swapped['players'][0]['matches'][0]['set_scores'], expected)
        # Each public view owns its array; editing one cannot corrupt another.
        recent['head_to_head'][0]['set_scores'][0]['winner_games'] = 99
        self.assertEqual(h2h['latest_meeting']['set_scores'], expected)
        self.assertEqual(loser_view['set_scores'], expected)
        self.assertEqual(row, original)

    def test_saved_history_game_scores_are_structured_in_mixed_recent_views(self):
        self.legacy[1]['score'] = '6-7(4) 6-3 6-2'
        original = copy.deepcopy(self.legacy)
        recent, _ = self.merge([api()])
        saved = next(row for row in recent['head_to_head'] if row['match_id'] == 'old2')
        expected = [
            {'winner_games':6, 'loser_games':7, 'tiebreak_loser_points':4, 'is_match_tiebreak':False},
            {'winner_games':6, 'loser_games':3, 'tiebreak_loser_points':None, 'is_match_tiebreak':False},
            {'winner_games':6, 'loser_games':2, 'tiebreak_loser_points':None, 'is_match_tiebreak':False},
        ]
        self.assertEqual(saved['set_scores'], expected)
        self.assertEqual(saved['winner']['id'], '2')
        for player in recent['players']:
            per_player = next(row for row in player['matches'] if row['match_id'] == 'old2')
            self.assertEqual(per_player['set_scores'], expected)
        self.assertEqual(self.legacy, original)

    def test_invalid_or_incomplete_game_details_stay_unknown_without_erasing_matches(self):
        for score in ('2-0', '6-4', '6-4 2-1', 'not recorded'):
            with self.subTest(score=score):
                recent, h2h = self.merge([api(score=score, score_kind='games')])
                self.assertEqual(recent['head_to_head'][0]['score'], score)
                self.assertEqual(recent['head_to_head'][0]['set_scores'], [])
                self.assertEqual(h2h['latest_meeting']['set_scores'], [])
                self.assertEqual(h2h['ytd']['played'], 1)

    def test_missing_surface_does_not_inherit_the_requested_surface(self):
        recent, h2h = self.merge([api(surface=None)])
        self.assertIsNone(recent['head_to_head'][0]['surface'])
        self.assertEqual(h2h['overall']['played'], 3)
        self.assertEqual(h2h['surface']['played'], 2)
        self.assertIn('without a recorded surface', h2h['coverage_note'])

    def test_no_feed_or_usable_records_does_not_erase_saved_results(self):
        for snapshot in (None, {}, {'records':[], 'status':{'coverage_complete':True,
                         'covered_ranges':[{'start':'2026-01-01', 'end':'2026-11-27'}]}}):
            with self.subTest(snapshot=snapshot):
                recent, h2h = self.merge([], feed_snapshot=snapshot)
                for field, value in self.old_recent.items():
                    self.assertEqual(recent[field], value)
                for field, value in self.old_h2h.items():
                    self.assertEqual(h2h[field], value)

    def test_inputs_are_not_mutated_and_returned_objects_do_not_share_input_state(self):
        snapshot = {'records':[api()], 'status':{'configured':True, 'coverage_complete':True,
                    'covered_ranges':[{'start':'2026-01-01', 'end':'2026-11-27'}],
                    'key':'private', 'authorization':'private'}}
        originals = copy.deepcopy((self.legacy, self.names, self.args, self.old_recent, self.old_h2h, snapshot))
        recent, h2h = self.merge([], feed_snapshot=snapshot)
        self.assertEqual((self.legacy, self.names, self.args, self.old_recent, self.old_h2h, snapshot), originals)
        self.assertNotIn('key', recent['feed'])
        self.assertNotIn('authorization', h2h['feed'])
        recent['head_to_head'][0]['winner']['name'] = 'Changed output'
        recent['feed']['covered_ranges'][0]['start'] = 'Changed output'
        self.assertEqual((self.legacy, self.names, self.args, self.old_recent, self.old_h2h, snapshot), originals)
        self.assertNotEqual(recent['feed'], h2h['feed'])


if __name__ == '__main__':
    unittest.main()
