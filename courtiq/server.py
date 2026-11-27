"""CourtIQ local website server. Python 3.9+, existing model dependencies.

Run from AI-Tennis-Match-Predictor: python run_courtiq.py
Serves only loopback; import CourtIQService in a production API deployment.
The exact model-training script and user's trained model remain unchanged.
"""
import argparse
from collections import OrderedDict
import copy
import csv
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import math
from pathlib import Path
import re
import socket
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit
import webbrowser

from courtiq.sportstats import SportStatsResults, parse_completed_score
from courtiq.history import merge_results_history
from courtiq.ui import load_html

HTML_TEMPLATE = load_html()
APP_VERSION = '1.5.0'
RATE_FIELDS = ('first_in','first_win','second_win','return1','return2','serve_won',
               'return_won','ace','double_fault','bp_save','bp_convert')
REPORTED_EVALUATION = {
    'validation_year': 2025, 'selection_year': 2024, 'features': 178,
    'model': {'matches':2275,'correct':1507,'accuracy':.6624175824175824,
              'log_loss':.6123123492864324,'brier':.2126840303739783,'auc':.7196514311236833},
    'elo': {'matches':2275,'correct':1453,'accuracy':.6386813186813187,
            'log_loss':.6304040453498765,'brier':.22024514796246528,'auc':.7049385243359263},
    'limitations': ['Historical evaluation reported by the project owner.',
                    'Results enter history after a 21-day tournament-date availability approximation.',
                    'The available ATP snapshot reaches tournament starts on May 25, 2026.']
}


def clean_number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def day_string(value):
    if value is None:
        return None
    return str(value)[:10]


class APIError(Exception):
    def __init__(self, message, code='invalid_request', status=400):
        super().__init__(message)
        self.code, self.status = code, status


def public_results_status(raw):
    """Present provider status without exposing transport details or secrets."""
    result = copy.deepcopy(raw)
    progress = result.get('progress')
    if isinstance(progress, dict):
        result['progress_details'] = progress
        result['progress'] = progress.get('percent', 0)
        result['message'] = progress.get('message', '')
    result['retry_after_seconds'] = result.get('delay_seconds', 0)
    result['delay_seconds'] = result.get('data_delay_seconds')
    return result


class CourtIQService:
    def __init__(self, root, model_path=None, source_path=None, results_provider=None):
        self.root = Path(root).resolve()
        preferred_model = self.root / 'models/courtiq/model.joblib'
        legacy_model = self.root / 'models/research_v5/model.joblib'
        self.model_path = Path(model_path) if model_path else (
            preferred_model if preferred_model.is_file() or not legacy_model.is_file() else legacy_model)
        self.source_path = Path(source_path) if source_path else Path(__file__).with_name('model.py')
        self.lock = threading.RLock()
        self.predictor, self.module, self.error = None, None, None
        self.cache = OrderedDict()
        self._players = []
        self._evaluation = None
        self.results_feed = results_provider if results_provider is not None else SportStatsResults(self.root)
        try:
            if not self.source_path.is_file():
                raise FileNotFoundError('The training script is missing. Run this website from your AI-Tennis-Match-Predictor project.')
            if not self.model_path.is_file():
                raise FileNotFoundError('The saved predictor is missing. Train V5 first, or pass --model with its model.joblib path.')
            spec = importlib.util.spec_from_file_location('courtiq_trained_model', self.source_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.module = module
            self.predictor = module.TennisPredictor.load(self.model_path)
            self._prepare_players()
            self._evaluation = self._read_evaluation()
        except (ValueError, FileNotFoundError, ImportError, OSError) as exc:
            self.error = str(exc)

    def _ready(self):
        if self.predictor is None:
            raise APIError(self.error or 'No trained model is connected.', 'model_unavailable', 503)

    def _dates(self):
        bundle = self.predictor.bundle
        minimum = max(day_string(bundle['trained_before']), day_string(bundle['engine_state']['current_date']))
        today = datetime.now(timezone.utc).date()
        default = max(minimum, today.isoformat())
        through = day_string(bundle['engine_state']['latest_known_tournament'])
        age = (today-date.fromisoformat(through)).days if through else None
        return minimum, default, through, age

    def status(self):
        if self.predictor is None:
            return {'ready':False,'app_version':APP_VERSION,'error':self.error or 'No model is connected.',
                    'notice':'Run the local app from the project containing your saved model.',
                    'results_feed':self.results_status()}
        b = self.predictor.bundle
        minimum, default, through, age = self._dates()
        return {'ready':True,'app_version':APP_VERSION,'model':{'name':'Research V5','mode':b['mode'],
                    'features':b['evaluation']['features'],'trained_before':b['trained_before'],
                    'data_through':through,'charting_enabled':bool(b['chart_provenance']['enabled'])},
                'default_date':default,'min_date':minimum,
                'max_date':(datetime.now(timezone.utc).date()+timedelta(days=366)).isoformat(),
                'source_age_days':age,'notice':'Historical model snapshot',
                'results_feed':self.results_status()}

    def results_status(self):
        return public_results_status(self.results_feed.status())

    def sync_results(self, payload):
        if not isinstance(payload, dict) or payload:
            raise APIError('Send an empty JSON object to refresh match results.')
        return public_results_status(self.results_feed.refresh(force=True))

    def _prepare_players(self):
        b = self.predictor.bundle
        self._players = []
        cutoff = self.module.tennis_date(self._dates()[1])
        history = b['engine_state']['history']
        metadata = b['engine_state']['metadata']
        for player, records in history.items():
            known = [r for r in records if r['completed'] and r['known_at'] < cutoff]
            if len(known) < 10:
                continue
            last = max(r['date'] for r in known)
            details = metadata.get(player, {})
            rank = clean_number(details.get('rank'))
            self._players.append({'id':str(player),'name':b['names'].get(str(player),str(player)),
                'last_rank':int(rank) if rank is not None else None,'last_match':day_string(last),
                'matches':len(known),'active_recently':(cutoff-last).days <= 730})
        self._players.sort(key=lambda p:(not p['active_recently'],p['last_rank'] or 100000,p['name']))

    def players(self):
        self._ready()
        return {'players':self._players}

    def _request(self, payload):
        self._ready()
        if not isinstance(payload,dict):
            raise APIError('Send a JSON object with two players and match conditions.')
        permitted = {'player_a','player_b','surface','event','level','best_of','round','as_of'}
        if set(payload)-permitted:
            raise APIError('Unrecognized matchup fields: '+', '.join(sorted(set(payload)-permitted)))
        for field in ('player_a','player_b'):
            if not isinstance(payload.get(field),str) or not 1 <= len(payload[field].strip()) <= 120:
                raise APIError('Select two players from the player list.')
        try:
            a,b=(self.predictor.resolve(payload[field].strip()) for field in ('player_a','player_b'))
        except ValueError as exc:
            raise APIError(str(exc))
        if a == b:
            raise APIError('Choose two different players.')
        surface = payload.get('surface','Hard')
        if surface not in ('Hard','Clay','Grass','Carpet'):
            raise APIError('Choose Hard, Clay, Grass or Carpet.')
        level = payload.get('level','A')
        if level not in ('A','M','G','F','D','O'):
            raise APIError('Choose a supported tournament level.')
        best = payload.get('best_of',3)
        if isinstance(best,bool) or not isinstance(best,int) or best not in (3,5):
            raise APIError('Match format must be best of 3 or best of 5.')
        # The streamlined website leaves the round unspecified. Its indicator
        # columns are all zero; do not silently invent a Round-of-32 matchup.
        round_name = payload.get('round','')
        if round_name not in ('','F','SF','QF','R16','R32','R64','R128','RR','BR'):
            raise APIError('Choose a supported tournament round.')
        event = payload.get('event','')
        if not isinstance(event,str) or len(event) > 120 or any(ord(c)<32 for c in event):
            raise APIError('Event name must be plain text of at most 120 characters.')
        minimum, default, _, _ = self._dates()
        forecast = payload.get('as_of',default)
        if not isinstance(forecast,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',forecast):
            raise APIError('Forecast date must use YYYY-MM-DD.')
        try:
            parsed = date.fromisoformat(forecast)
        except ValueError:
            raise APIError('Choose a valid calendar date.')
        if forecast < minimum:
            raise APIError('This app uses the saved snapshot available from %s. Choose that date or later.'%minimum)
        if parsed > datetime.now(timezone.utc).date()+timedelta(days=366):
            raise APIError('Choose a forecast date within the next year.')
        return {'player_a':a,'player_b':b,'surface':surface,'event':event.strip(),
                'level':level,'best_of':best,'round':round_name,'as_of':forecast}

    def _stats(self, player, surface, asof):
        engine = self.predictor.engine
        snapshot = engine.sipko.snapshot(player,surface,asof)
        records = [r for r in engine.history[player] if r['completed'] and r['known_at'] < asof]
        if len(records) < 10:
            raise APIError('Both players need at least 10 available completed matches.')
        recent = records[-10:]
        window = [r for r in records if (asof-r['date']).days <= self.predictor.bundle['config']['history_days']]
        same = [r for r in window if r['surface'] == surface]
        meta = engine.metadata.get(player,{})
        rates = {}
        for field in RATE_FIELDS:
            opportunities = clean_number(snapshot.get(field+'_opportunities')) or 0.
            rates[field] = {'value':clean_number(snapshot.get(field)) if opportunities > 0 else None,
                            'weighted_opportunities':opportunities}
        rank = clean_number(meta.get('rank'))
        last = records[-1] if records else None
        return {'id':player,'name':self.predictor.bundle['names'][player],'rates':rates,
                'recent_form':{'played':len(recent),'wins':int(sum(r['win'] for r in recent)),
                               'win_rate':sum(r['win'] for r in recent)/len(recent) if recent else None},
                'surface_history':{'surface':surface,'played':len(same),'wins':int(sum(r['win'] for r in same)),
                                   'win_rate':sum(r['win'] for r in same)/len(same) if same else None},
                'history':{'window_matches':len(window),'last_tournament_start':day_string(last['date']) if last else None,
                           'last_known_at':day_string(last['known_at']) if last else None},
                'rank':{'last_recorded':int(rank) if rank is not None else None,
                        'as_of_tournament_start':day_string(meta.get('date'))}}

    def _comparison(self, args):
        asof = self.module.tennis_date(args['as_of'])
        self.predictor._state(asof)
        through = day_string(self.predictor.engine.latest_known_tournament)
        feed = self.results_feed.snapshot()
        feed['status'] = public_results_status(feed.get('status', {}))
        recent, head_to_head = merge_results_history(
            self.predictor.records, self.predictor.bundle['names'], args, through,
            self._recent_matches(args), self._head_to_head(args), feed)
        return {'as_of':args['as_of'],'surface':args['surface'],
                'players':[self._stats(args[k],args['surface'],asof) for k in ('player_a','player_b')],
                'data_through':through,
                'head_to_head':head_to_head,
                'recent_matches':recent}

    def _recent_matches(self, args):
        """Descriptive completed results from all saved history and all courts.

        Tournament starts are date proxies, not verified match timestamps. Within
        a date, tournament ID, match number and match ID provide a stable order.
        These records do not change any features or prediction probabilities.
        """
        a, b = args['player_a'], args['player_b']
        asof = self.module.tennis_date(args['as_of'])
        names = self.predictor.bundle['names']
        eligible, seen = [], set()
        for row in self.predictor.records:
            if not row['completed'] or not row['known_at'] < asof:
                continue
            if not {a,b}.intersection((row['winner_id'],row['loser_id'])):
                continue
            key = row['match_id']
            if key not in seen:
                eligible.append(row)
                seen.add(key)
        eligible.sort(key=lambda r:(r['date'],str(r['tourney_id']),
                                    clean_number(r.get('match_num')) or 0.,str(r['match_id'])),
                      reverse=True)

        def optional_text(value):
            if value is None:
                return None
            text = str(value).strip()
            return text if text.lower() not in ('nan','none','nat','<na>','') else None

        def record(row):
            winner, loser = row['winner_id'], row['loser_id']
            score = optional_text(row.get('score'))
            sets = parse_completed_score(score)
            return {'match_id':str(row['match_id']),'tournament_date':day_string(row['date']),
                    'winner':{'id':winner,'name':names.get(winner,optional_text(row.get('winner_name')) or winner)},
                    'loser':{'id':loser,'name':names.get(loser,optional_text(row.get('loser_name')) or loser)},
                    'event':optional_text(row.get('tourney_name')),
                    'surface':optional_text(row.get('surface')),'score':score,
                    'score_kind':'games' if sets else 'unknown','set_scores':sets}

        meetings = [r for r in eligible if {r['winner_id'],r['loser_id']} == {a,b}]
        players = []
        for player in (a,b):
            history = [r for r in eligible if player in (r['winner_id'],r['loser_id'])][:3]
            matches = []
            for row in history:
                item = record(row)
                won = row['winner_id'] == player
                item.update(result='W' if won else 'L',opponent=dict(item['loser' if won else 'winner']))
                matches.append(item)
            players.append({'id':player,'name':names[player],'matches':matches})
        return {'head_to_head':[record(row) for row in meetings[:3]],'players':players,
                'as_of':args['as_of'],'data_through':day_string(self.predictor.engine.latest_known_tournament),
                'source':'saved_history','date_basis':'tournament_start','score_basis':'winner_first',
                'scope':'Up to three latest recorded completed matches across all surfaces, from the full saved history. '
                        'Retirements, walkovers, defaults and abandoned matches are excluded. Only records released before '
                        'the forecast date are included. Dates are tournament starts; ties use tournament and match identifiers, '
                        'not verified match times. Scores are recorded in winner-first order.'}

    def _head_to_head(self, args):
        """Raw recorded meetings across the full saved history, not the feature window."""
        a, b = args['player_a'], args['player_b']
        asof = self.module.tennis_date(args['as_of'])
        meetings = []
        seen = set()
        for row in self.predictor.records:
            if not row['completed'] or not row['known_at'] < asof:
                continue
            if not ((row['winner_id'] == a and row['loser_id'] == b) or
                    (row['winner_id'] == b and row['loser_id'] == a)):
                continue
            key = row['match_id']
            if key not in seen:
                meetings.append(row)
                seen.add(key)
        meetings.sort(key=lambda r:(r['date'], str(r['tourney_id']), float(r.get('match_num',0))))

        def totals(rows):
            wins_a = sum(row['winner_id'] == a for row in rows)
            return {'played':len(rows),'wins_a':int(wins_a),'wins_b':len(rows)-int(wins_a)}

        names = self.predictor.bundle['names']
        latest = meetings[-1] if meetings else None
        # A blank score/event is missing information, never a fabricated value.
        def optional_text(value):
            return str(value) if value is not None and str(value).lower() not in ('nan','none','nat','') else None

        return {'players':[{'id':a,'name':names[a]},{'id':b,'name':names[b]}],
                'year':int(asof.year),
                'ytd':totals([r for r in meetings if r['date'].year == asof.year]),
                'overall':totals(meetings),
                'surface':dict(totals([r for r in meetings if r['surface'] == args['surface']]),surface=args['surface']),
                'data_through':day_string(self.predictor.engine.latest_known_tournament),
                'history_start':day_string(min(r['date'] for r in self.predictor.records)) if self.predictor.records else None,
                'latest_meeting':{'date':day_string(latest['date']),
                    'event':optional_text(latest.get('tourney_name')),'surface':latest['surface'],
                    'winner_id':latest['winner_id'],'winner_name':names[latest['winner_id']],
                    'score':optional_text(latest.get('score')),
                    'set_scores':parse_completed_score(optional_text(latest.get('score')))} if latest else None,
                'scope':'Completed matches in the saved dataset across all surfaces. Retirements, walkovers, defaults and abandoned matches are excluded. Year is based on tournament-start date; only released records are counted.'}

    def _prediction_math(self, args, expected_probability):
        """Expose exact model components, not a causal feature-importance claim."""
        module, predictor = self.module, self.predictor
        engine, estimator = predictor.engine, predictor.bundle['estimator']
        a, b = args['player_a'], args['player_b']
        common = dict(surface=args['surface'], asof=module.tennis_date(args['as_of']),
                      event=args['event'],level=args['level'],best_of=args['best_of'],round_name=args['round'])
        forward = engine.matchup(a,b,**common)
        reverse = engine.matchup(b,a,**common)
        X = module._pe_frame(module.pd.DataFrame([forward]),estimator['features'])
        Xswap = module._pe_frame(module.pd.DataFrame([reverse]),estimator['features'])
        weight = float(estimator['catboost_weight'])
        components = {}
        for kind, fitted in estimator['models'].items():
            if kind == 'logistic':
                pf = float(module._pe_logistic_probability(fitted,X)[0])
                pr = float(module._pe_logistic_probability(fitted,Xswap)[0])
                component_weight = 1-weight
            elif kind == 'catboost':
                pf, pr = float(fitted.predict_proba(X)[0,1]), float(fitted.predict_proba(Xswap)[0,1])
                component_weight = weight
            else:
                continue
            components[kind] = {'forward':pf,'reverse':pr,'symmetric':(pf+1-pr)/2,'weight':component_weight}
        reconstructed = sum(v['symmetric']*v['weight'] for v in components.values())
        if not math.isclose(reconstructed,expected_probability,abs_tol=1e-12,rel_tol=1e-12):
            raise RuntimeError('Model explanation did not reconstruct the prediction.')
        ea, eb = float(engine.elo[a]), float(engine.elo[b])
        sa, sb = float(engine.surface_elo[a,args['surface']]), float(engine.surface_elo[b,args['surface']])
        return {'source':'saved_model','features_count':len(estimator['features']),
                'round_policy':'unspecified' if not args['round'] else 'provided',
                'components':components,'probability_a':reconstructed,'probability_b':1-reconstructed,
                'elo':{'a':ea,'b':eb,'probability_a':1/(1+10**((eb-ea)/400)),
                       'surface_a':sa,'surface_b':sb,'surface_probability_a':1/(1+10**((sb-sa)/400))},
                'feature_values':{name:clean_number(forward[name]) for name in estimator['features']},
                'note':'These are the exact component probabilities for this matchup. Feature values and component weights are not causal explanations or feature importance.'}

    def compare(self, payload):
        with self.lock:
            args = self._request(payload)
            return self._comparison(args)

    def predict(self, payload, progress=None):
        """Compute the unchanged response, optionally reporting real work milestones.

        Percentages are coarse pipeline stages, never an estimate of elapsed time.
        A disconnected streaming client can stop at a callback boundary without
        changing the model, cached results, or the next request.
        """
        def update(stage, message, percent):
            if progress is not None:
                progress({'type':'progress','stage':stage,'message':message,'progress':percent})

        update('queued','Request received. Waiting for the prediction engine.',4)
        with self.lock:
            update('validating','Checking players and match conditions.',8)
            args = self._request(payload)
            update('validated','Players and match conditions checked.',12)
            # Result history has its own revision. A feed refresh invalidates
            # descriptive output without changing the trained model or inputs.
            feed_revision = self.results_status().get('revision', 0)
            key = tuple(args.items()) + (('_results_revision', feed_revision),)
            if key in self.cache:
                self.cache.move_to_end(key)
                result=copy.deepcopy(self.cache[key])
                result['cached']=True
                update('cache_hit','Reusing the saved result for this exact matchup.',95)
                update('complete','Prediction, player profiles and head-to-head ready.',100)
                return result
            try:
                update('predicting','Building historical matchup features and running the saved model.',18)
                prediction=self.predictor.predict(args['player_a'],args['player_b'],args['surface'],args['as_of'],
                    event=args['event'],level=args['level'],best_of=args['best_of'],round_name=args['round'])
                update('comparison','Calculating historical player rates and recorded head-to-head.',62)
                comparison=self._comparison(args)
                update('explanation','Checking model components in both player orders.',84)
                math_details=self._prediction_math(args,prediction['probability_a'])
            except ValueError as exc:
                raise APIError(str(exc))
            result={'prediction':prediction,'comparison':comparison,
                    'head_to_head':comparison['head_to_head'],'recent_matches':comparison['recent_matches'],
                    'math':math_details,'cached':False}
            self.cache[key]=copy.deepcopy(result)
            while len(self.cache)>128:
                self.cache.popitem(last=False)
            update('complete','Prediction, player profiles and head-to-head ready.',100)
            return result

    def _read_evaluation(self):
        report=copy.deepcopy(self.predictor.bundle['evaluation'])
        calibration=[]
        file=self.model_path.parent/'validation_predictions.csv'
        if file.is_file():
            try:
                with file.open(newline='') as handle:
                    rows=list(csv.DictReader(handle))
                values=[(float(r['probability_a']),int(r['outcome_a'])) for r in rows]
                valid=(len(values)==report['model']['matches'] and
                       all(math.isfinite(p) and 0<=p<=1 and y in (0,1) for p,y in values) and
                       sum(int((p>=.5)==y) for p,y in values)==report['model']['correct'])
                if valid:
                    for index in range(10):
                        selected=[(p,y) for p,y in values if min(int(p*10),9)==index]
                        if selected:
                            calibration.append({'low':index/10,'high':(index+1)/10,'count':len(selected),
                                'predicted':sum(p for p,y in selected)/len(selected),
                                'observed':sum(y for p,y in selected)/len(selected)})
            except (ValueError,KeyError,OSError):
                calibration=[]
        label='%s historical evaluation'%report['validation_year']
        if self.predictor.bundle['mode']=='deployment_refit':
            label+=' · earlier evaluated model'
        return {'report':report,'calibration':calibration,'source':'saved_model','evaluation_label':label}

    def evaluation(self):
        if self.predictor is None:
            return {'report':copy.deepcopy(REPORTED_EVALUATION),'calibration':[],
                    'source':'reported_run','evaluation_label':'2025 historical evaluation · reported run'}
        return copy.deepcopy(self._evaluation)

    def methodology(self):
        """Configuration from saved weights, with a clearly marked reported fallback."""
        config = {'training_start':2010,'training_half_life_years':5,
                  'history_discount':.8,'history_days':1826,'prior_points':50.,
                  'surface_prior':.25,'surface_shrink_players':20.,'surface_refresh_days':30.,
                  'availability_delay_days':21,'catboost_weight':.25,'logistic_weight':.75,
                  'selection_year':2024,'features':178,'round_policy':'unspecified',
                  'elo_initial':1500.,'elo_k':24.,'elo_scale':400.}
        feature_names, groups, interactions, details = [], [], [], {}
        source = 'reported_run'
        if self.predictor is not None:
            bundle = self.predictor.bundle
            estimator = bundle['estimator']
            sipko = bundle['engine_state']['sipko']
            config.update(bundle['config'])
            config.update(training_start=estimator['training_config']['start'],
                          training_half_life_years=estimator['training_config']['half_life'],
                          catboost_weight=float(estimator['catboost_weight']),
                          logistic_weight=1-float(estimator['catboost_weight']),
                          selection_year=estimator['audit']['selection_year'],
                          features=len(estimator['features']),
                          surface_prior=float(sipko['surface_prior']),
                          surface_shrink_players=float(sipko['surface_shrink']),
                          surface_refresh_days=float(sipko['refresh'])/86400)
            feature_names=list(estimator['features'])
            grouped={}
            for name in feature_names:
                group=('Match context' if name.startswith('context_') else
                       'History coverage' if name.startswith('coverage_') else
                       'Player differences' if name.startswith('diff_') else 'Other derived features')
                grouped[group]=grouped.get(group,0)+1
            groups=[{'name':name,'count':count} for name,count in grouped.items()]
            interactions=[list(p) for p in estimator['interaction_pairs']]
            logistic=estimator['models'].get('logistic')
            catboost=estimator['models'].get('catboost')
            details={'logistic_active':logistic is not None,'catboost_active':catboost is not None,
                     'interaction_count':len(interactions),
                     'logistic_coefficient_count':int(logistic['classifier'].coef_.shape[1]) if logistic else None,
                     'catboost_trees':int(catboost.tree_count_) if catboost is not None else None,
                     'catboost_depth':int(catboost.get_params().get('depth',5)) if catboost is not None else None,
                     'calibration':'No separate probability calibrator is fitted; the plot evaluates calibration.'}
            source='saved_model'
        priors = dict(self.module._SF_PRIOR_RATES) if self.module is not None else {
            'first_in':.62,'first_win':.72,'second_win':.51,'return1':.28,'return2':.49,
            'serve_won':.64,'return_won':.36,'ace':.08,'double_fault':.035,'bp_save':.60,'bp_convert':.40}
        return {'source':source,'config':config,'feature_groups':groups,'features':feature_names,
                'interactions':interactions,'models':details,'prior_rates':priors,
                'round_note':'Round is not supplied by this interface; its round indicator features are zero. Historical evaluation used recorded match context.',
                'date_note':'The interface uses the server UTC calendar date automatically, bounded by the saved snapshot. Forecast date does not refresh the underlying data.'}


def make_handler(service, html=None, html_path=None):
    class Handler(BaseHTTPRequestHandler):
        server_version='CourtIQ'

        def log_message(self, fmt, *args):
            if self.path.startswith('/api/predict') or ' 5' in (fmt%args):
                print('[CourtIQ] '+fmt%args,flush=True)

        def _trusted_request(self):
            port=self.server.server_address[1]
            allowed={'127.0.0.1:%s'%port,'localhost:%s'%port}
            host=self.headers.get('Host','').lower()
            if host not in allowed:
                raise APIError('This local server accepts localhost requests only.','invalid_host',403)
            origin=self.headers.get('Origin')
            if origin:
                parsed=urlsplit(origin)
                if parsed.scheme!='http' or parsed.netloc.lower() not in allowed:
                    raise APIError('Requests from other websites are not accepted.','invalid_origin',403)

        def _send(self, status, payload, content_type='application/json; charset=utf-8'):
            body=json.dumps(payload,allow_nan=False).encode() if isinstance(payload,(dict,list)) else payload
            if isinstance(body,str):body=body.encode()
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Referrer-Policy','same-origin')
            self.end_headers()
            self.wfile.write(body)

        def _error(self, exc):
            self._send(exc.status,{'error':{'code':exc.code,'message':str(exc)}})

        def _stream_predict(self, payload):
            # The local server uses a close-delimited NDJSON response. Each
            # milestone is flushed when its corresponding computation occurs.
            self.send_response(200)
            self.send_header('Content-Type','application/x-ndjson; charset=utf-8')
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Referrer-Policy','same-origin')
            self.send_header('X-Accel-Buffering','no')
            self.send_header('Connection','close')
            self.end_headers()
            self.close_connection=True

            def emit(event):
                body=(json.dumps(event,allow_nan=False,separators=(',',':'))+'\n').encode('utf-8')
                self.wfile.write(body)
                self.wfile.flush()

            try:
                result=service.predict(payload,progress=emit)
                emit({'type':'result','data':result})
            except APIError as exc:
                emit({'type':'error','error':str(exc),'code':exc.code})
            except (BrokenPipeError,ConnectionResetError,socket.timeout):
                pass
            except Exception as exc:
                print('Streaming prediction failed: '+str(exc),file=sys.stderr)
                emit({'type':'error','error':'Prediction could not be completed. Check the local server terminal.',
                      'code':'server_error'})

        def do_GET(self):
            try:
                self._trusted_request()
                parsed=urlsplit(self.path)
                if parsed.path in ('/','/index.html'):
                    content=Path(html_path).read_text() if html_path and Path(html_path).is_file() else html or HTML_TEMPLATE
                    self._send(200,content,'text/html; charset=utf-8')
                elif parsed.path=='/api/status':self._send(200,service.status())
                elif parsed.path=='/api/results/status':self._send(200,service.results_status())
                elif parsed.path=='/api/players':self._send(200,service.players())
                elif parsed.path=='/api/evaluation':self._send(200,service.evaluation())
                elif parsed.path=='/api/methodology':self._send(200,service.methodology())
                elif parsed.path=='/api/compare':
                    query=parse_qs(parsed.query,keep_blank_values=True)
                    if any(len(v)!=1 for v in query.values()):raise APIError('Each query field may appear only once.')
                    payload={k:v[0] for k,v in query.items()}
                    if 'best_of' in payload:
                        if payload['best_of'] not in ('3','5'):raise APIError('Match format must be best of 3 or best of 5.')
                        payload['best_of']=int(payload['best_of'])
                    self._send(200,service.compare(payload))
                elif parsed.path=='/favicon.ico':self._send(204,b'','image/x-icon')
                else:raise APIError('Page not found.','not_found',404)
            except APIError as exc:self._error(exc)
            except (BrokenPipeError,ConnectionResetError):pass
            except Exception as exc:
                print('Request failed: '+str(exc),file=sys.stderr)
                self._error(APIError('The request could not be completed. Check the local server terminal.','server_error',500))

        def do_POST(self):
            try:
                self._trusted_request()
                endpoint=urlsplit(self.path).path
                if endpoint not in ('/api/predict','/api/predict/stream','/api/results/sync'):raise APIError('Endpoint not found.','not_found',404)
                if self.headers.get('Transfer-Encoding'):raise APIError('Chunked requests are not supported.','invalid_body',400)
                if self.headers.get_content_type()!='application/json':raise APIError('Send application/json.','invalid_content_type',415)
                try:length=int(self.headers.get('Content-Length','0'))
                except ValueError:raise APIError('Invalid request length.')
                if not 0<length<=8192:raise APIError('Request body must be between 1 and 8192 bytes.','invalid_body',413)
                self.connection.settimeout(15)
                try:payload=json.loads(self.rfile.read(length).decode('utf-8'))
                except (ValueError,UnicodeError):raise APIError('Request body must contain valid JSON.')
                if endpoint=='/api/results/sync':self._send(200,service.sync_results(payload))
                elif endpoint=='/api/predict/stream':self._stream_predict(payload)
                else:self._send(200,service.predict(payload))
            except APIError as exc:self._error(exc)
            except (BrokenPipeError,ConnectionResetError):pass
            except socket.timeout:self._error(APIError('The request body timed out.','timeout',408))
            except Exception as exc:
                print('Prediction failed: '+str(exc),file=sys.stderr)
                self._error(APIError('Prediction could not be completed. Check the local server terminal.','server_error',500))
    return Handler


def main(argv=None):
    parser=argparse.ArgumentParser(description='Run the CourtIQ website with your existing trained tennis model.')
    parser.add_argument('--version',action='version',version='CourtIQ '+APP_VERSION)
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]),help='AI-Tennis-Match-Predictor project folder.')
    parser.add_argument('--model',help='Optional model path; defaults to models/courtiq/model.joblib, with legacy research_v5 fallback.')
    parser.add_argument('--model-code','--source',dest='model_code',help='Exact model source used during training; defaults to courtiq/model.py.')
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--no-browser',action='store_true')
    parser.add_argument('--html',help='Optional editable frontend HTML file.')
    parser.add_argument('--export-ui',metavar='PATH',help='Export the assembled frontend for editing, then exit.')
    args=parser.parse_args(argv)
    if args.export_ui:
        out=Path(args.export_ui)
        if out.exists():parser.error('Choose a new filename; this export preserves existing files.')
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(HTML_TEMPLATE)
        print('Saved editable frontend: '+str(out))
        return
    if not 1024<=args.port<=65535:parser.error('Choose a port from 1024 to 65535.')
    print('CourtIQ %s · Loading your saved tennis predictor...'%APP_VERSION,flush=True)
    service=CourtIQService(args.root,args.model,args.model_code)
    try:server=ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(service,html_path=args.html))
    except OSError as exc:parser.exit(2,'Could not start local website: %s. Try --port 8001.\n'%exc)
    server.daemon_threads=True
    results_stop=threading.Event()
    def periodic_results():
        while not results_stop.wait(6*60*60):
            service.results_feed.refresh()
    url='http://127.0.0.1:%s'%args.port
    if service.predictor is None:print('Model setup: '+str(service.error),flush=True)
    else:print('Model ready. %s players available.'%len(service._players),flush=True)
    if service.results_status().get('configured'):
        service.results_feed.refresh()
        threading.Thread(target=periodic_results,name='courtiq-results-schedule',daemon=True).start()
        print('SportStatsAPI configured. Results sync runs in the background; model predictions remain available.',flush=True)
    print('CourtIQ is running at '+url+'\nKeep this terminal open. Press Control-C to stop.',flush=True)
    if not args.no_browser:threading.Timer(.5,lambda:webbrowser.open(url)).start()
    try:server.serve_forever()
    except KeyboardInterrupt:print('\nCourtIQ stopped.')
    finally:
        results_stop.set()
        service.results_feed.close()
        server.server_close()


if __name__=='__main__':main()
