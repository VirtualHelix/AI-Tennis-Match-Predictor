#!/usr/bin/env python3
"""Tennis research model: integrated training and named-player prediction.

Requires Python 3.9+, numpy, pandas, scikit-learn, catboost and joblib.
This is a single file: no imports from earlier experiment scripts are needed.

TRAIN
  python tennis_research_model.py train
PREDICT
  python tennis_research_model.py predict --player-a "Jannik Sinner" \
    --player-b "Carlos Alcaraz" --surface Hard --date 2026-11-26
DEPLOYMENT REFIT (optional, preserves the evaluated model)
  python tennis_research_model.py finalize --as-of 2026-11-26

METHOD PROVENANCE
Sipko, Machine Learning for the Prediction of Professional Tennis Matches
(2015), printed pp.14-21: common opponents, exact Eq.3.2 time discount,
surface similarity, WSP and COMPLETE. Online surface estimation, pooled
counts, shrinkage and the chosen interactions are documented adaptations.
Lei et al., Rhythms of Victory (IEEE Access 2024), DOI
10.1109/ACCESS.2024.3444031: motivates dynamic momentum. This implementation
adapts the idea to pre-match residuals of completed matches. The paper's
83.52% next-point result is NOT a claimed pre-match accuracy here.
Sampaio et al., Applied Sciences 14,5517 (2024), DOI10.3390/app14135517:
motivates serve/return, shot information, data quality and longitudinal
assessment; it does not prescribe a single winning predictive model.

MODEL SCOPE
Default training starts2010 with five-year row-weight half-life. The fixed
CatBoost/logistic mixture is chosen on2024, fitted through2024, and reported
on2025. No2026 score is produced by train. Training and serving use the same
causal feature code and saved input snapshot. Results enter history only
after tournament-start+21days, an explicit approximation. Charted shots
require verified availability or explicit --research-charting. No odds,
trading, paper trading, psychological ratings or invented accuracy claims.
"""



# ==================== sipko_features.py ====================
"""Past-only Sipko-inspired features for CourtIQ (Python 3.9+).

Sipko (2015), printed pp. 14--20, motivates common-opponent comparisons,
surface similarity, W(t)=min(f**t, f), WSP and COMPLETE. The decay equation
is reproduced exactly, with t in 365.25-day years. Count pooling, fixed
priors, reliability shrinkage, monthly online surface re-estimation, and
stored pre-match opponent residuals are explicit adaptations, not a
replication or a claim that these features improve accuracy.

Records require date, known_at, surface, opponent, win. Serve fields are
svpt, first_in_count, first_won_count, second_won_count, aces,
double_faults, bp_saved, bp_faced. Return fields are return_svpt,
return_first_in_count, return_first_won_count, return_second_won_count,
opp_bp_saved, opp_bp_faced. Return *_won_count fields mean points WON BY
THE OPPONENT ON SERVE, as in ATP data. Optional adj_serve, adj_return and
win_residual must already be computed against BEFORE-MATCH expectations.

Only date <= known_at < asof is visible. Dates without time zones are UTC
date proxies, not asserted exact match times. Supplying an honest known_at
is the caller's responsibility. No target-row statistics enter a query.
"""
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import date, datetime, timezone
import math
_SF_DAY = 86400.0
_SF_YEAR = 365.25 * _SF_DAY
_SF_SURFACES = ('hard', 'clay', 'grass', 'carpet')
_SF_PRIOR_RATES = {'first_in': 0.62, 'first_win': 0.72, 'second_win': 0.51, 'return1': 0.28, 'return2': 0.49, 'serve_won': 0.64, 'return_won': 0.36, 'ace': 0.08, 'double_fault': 0.035, 'bp_save': 0.6, 'bp_convert': 0.4}
_SF_RESIDUALS = ('adj_serve', 'adj_return', 'win_residual')

def _sf_number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None

def _sf_stamp(value):
    try:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, date):
            parsed = datetime.combine(value, datetime.min.time())
        else:
            text = str(value).strip()
            if text.endswith('.0'):
                text = text[:-2]
            parsed = datetime.strptime(text, '%Y%m%d') if len(text) == 8 and text.isdigit() else datetime.fromisoformat(text.replace('Z', '+00:00'))
        return _sf_number((parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp())
    except (TypeError, ValueError, OverflowError):
        return None

def _sf_identity(value):
    if value is None or str(value).strip().casefold() in ('', 'nan', 'none', 'nat'):
        raise ValueError('player and opponent identities must be nonmissing')
    return str(value).strip()

def _sf_surface(value):
    text = str(value).strip().casefold()
    return 'unknown' if text in ('', 'nan', 'none', 'nat') else text

def _sf_positive(name, value):
    number = _sf_number(value)
    if number is None or number <= 0:
        raise ValueError(name + ' must be finite and positive')
    return number

def _sf_temporal_weight(age_years, discount_factor=0.8):
    """Sipko equation 3.2: the previous year has constant weight f."""
    f, age = (_sf_number(discount_factor), _sf_number(age_years))
    if f is None or not 0 < f <= 1 or age is None or (age < 0):
        raise ValueError('Require 0 < discount_factor <= 1 and nonnegative age_years')
    return min(f ** age, f)

def _sf_pair(numerator, denominator):
    n, d = (_sf_number(numerator), _sf_number(denominator))
    return (n, d) if n is not None and d is not None and (d > 0) and (0 <= n <= d) else None

def _sf_service(record, returning=False):
    prefix = 'return_' if returning else ''
    s, f, w1, w2 = (_sf_number(record.get(prefix + field)) for field in ('svpt', 'first_in_count', 'first_won_count', 'second_won_count'))
    bad = s is not None and s < 0 or (f is not None and f < 0) or (s is not None and f is not None and (f > s)) or (w1 is not None and (w1 < 0 or (f is not None and w1 > f))) or (w2 is not None and (w2 < 0 or (s is not None and f is not None and (w2 > s - f))))
    if not returning:
        df, aces = (_sf_number(record.get('double_faults')), _sf_number(record.get('aces')))
        bad |= df is not None and (df < 0 or (s is not None and f is not None and (df > s - f)))
        bad |= all((v is not None for v in (s, f, w2, df))) and w2 + df > s - f
        bad |= aces is not None and (aces < 0 or (w1 is not None and w2 is not None and (aces > w1 + w2)))
    return (None, None, None, None) if bad else (s, f, w1, w2)

def _sf_counts(record):
    s, f, w1, w2 = _sf_service(record)
    rs, rf, rw1, rw2 = _sf_service(record, returning=True)
    second = s - f if s is not None and f is not None else None
    rsecond = rs - rf if rs is not None and rf is not None else None
    total = w1 + w2 if w1 is not None and w2 is not None else None
    returns = rs - rw1 - rw2 if all((v is not None for v in (rs, rw1, rw2))) else None
    faced, saved = (_sf_number(record.get('opp_bp_faced')), _sf_number(record.get('opp_bp_saved')))
    return {'first_in': _sf_pair(f, s), 'first_win': _sf_pair(w1, f), 'second_win': _sf_pair(w2, second), 'return1': _sf_pair(rf - rw1 if rf is not None and rw1 is not None else None, rf), 'return2': _sf_pair(rsecond - rw2 if rsecond is not None and rw2 is not None else None, rsecond), 'serve_won': _sf_pair(total, s), 'return_won': _sf_pair(returns, rs), 'ace': _sf_pair(record.get('aces'), s), 'double_fault': _sf_pair(record.get('double_faults'), s), 'bp_save': _sf_pair(record.get('bp_saved'), record.get('bp_faced')), 'bp_convert': _sf_pair(faced - saved if faced is not None and saved is not None else None, faced)}

def observed_rates(record):
    """Validated completed-match rates for caller-computed opponent residuals.

    These are outcomes and must only be stored for release after the match;
    they must never be used as that match's predictive features.
    """
    pairs = _sf_counts(record)
    return {key: pairs[key][0] / pairs[key][1] if pairs[key] else math.nan for key in ('serve_won', 'return_won')}

class SipkoFeatures:
    """Serializable state with causal, out-of-order-safe queries.

    Surface correlations use player win rates available BEFORE each fixed
    30-day calendar bucket, with at least min_surface_matches per surface
    and min_shared_players in common. Correlations shrink toward an explicit
    .25 off-diagonal prior and are clipped to [0,1]. Correlations reflect
    observed schedule/opponent mix and are an estimate, not a physical
    measure of court similarity. Unknown surfaces use the fixed prior.
    """

    def __init__(self, discount_factor=0.8, history_days=1826, prior_points=50, min_surface_matches=8, min_shared_players=8, surface_refresh_days=30, surface_shrink_players=20, other_surface_prior=0.25, prior_matches=5, prior_opponents=5):
        _sf_temporal_weight(0, discount_factor)
        self.discount_factor = float(discount_factor)
        self.history_days = _sf_positive('history_days', history_days)
        self.prior_points = _sf_positive('prior_points', prior_points)
        self.min_surface_matches = _sf_positive('min_surface_matches', min_surface_matches)
        self.min_shared_players = max(3, int(_sf_positive('min_shared_players', min_shared_players)))
        self.refresh = _sf_positive('surface_refresh_days', surface_refresh_days) * _SF_DAY
        self.surface_shrink = _sf_positive('surface_shrink_players', surface_shrink_players)
        self.prior_matches = _sf_positive('prior_matches', prior_matches)
        self.prior_opponents = _sf_positive('prior_opponents', prior_opponents)
        p = _sf_number(other_surface_prior)
        if p is None or not 0 <= p <= 1:
            raise ValueError('other_surface_prior must be in [0,1]')
        self.surface_prior = p
        self.records, self.times = (defaultdict(list), defaultdict(list))
        self.seen, self._surface_cache, self._profile_cache = ({}, {}, {})
        self._profile_time = None

    def update(self, player_id, record):
        """Store one completed player's result; match_id makes repeats idempotent."""
        player, opponent = (_sf_identity(player_id), _sf_identity(record.get('opponent')))
        if player == opponent:
            raise ValueError('A player cannot play themselves')
        played, known, win = (_sf_stamp(record.get('date')), _sf_stamp(record.get('known_at')), _sf_number(record.get('win')))
        if played is None or known is None or played > known or (win not in (0, 1)):
            raise ValueError('Require valid date <= known_at and win in {0,1}')
        residuals = {}
        for name in _SF_RESIDUALS:
            value = _sf_number(record.get(name))
            if value is not None and (not -1 <= value <= 1):
                raise ValueError(name + ' must be in [-1,1]')
            residuals[name] = value
        item = {'date': played, 'known': known, 'surface': _sf_surface(record.get('surface')), 'opponent': opponent, 'win': win, 'counts': _sf_counts(record), 'residuals': residuals}
        if record.get('match_id') is not None:
            key = (player, str(record['match_id']))
            if key in self.seen:
                if item != self.seen[key]:
                    raise ValueError('Conflicting update for an existing player/match_id')
                return False
            self.seen[key] = item
        position = bisect_right(self.times[player], known)
        self.times[player].insert(position, known)
        self.records[player].insert(position, item)
        for cutoff in list(self._surface_cache):
            if known < cutoff:
                del self._surface_cache[cutoff]
        self._profile_cache.clear()
        return True

    def _past(self, player, now):
        times = self.times.get(player, ())
        lo = bisect_left(times, now - self.history_days * _SF_DAY)
        hi = bisect_left(times, now)
        return [r for r in self.records.get(player, ())[lo:hi] if 0 <= now - r['date'] <= self.history_days * _SF_DAY]

    def _surface_state(self, now):
        cutoff = math.floor(now / self.refresh) * self.refresh
        if cutoff in self._surface_cache:
            return self._surface_cache[cutoff]
        rates = {s: {} for s in _SF_SURFACES}
        for player in self.records:
            totals = defaultdict(lambda: [0.0, 0.0, 0])
            for r in self._past(player, cutoff):
                if r['surface'] in rates:
                    w = _sf_temporal_weight((cutoff - r['date']) / _SF_YEAR, self.discount_factor)
                    t = totals[r['surface']]
                    t[0] += w * r['win']
                    t[1] += w
                    t[2] += 1
            for s, (wins, weight, count) in totals.items():
                if count >= self.min_surface_matches:
                    rates[s][player] = (wins + 0.5 * self.prior_matches) / (weight + self.prior_matches)
        matrix, coverage = ({}, {})
        for a in _SF_SURFACES:
            matrix[a], coverage[a] = ({}, {})
            for b in _SF_SURFACES:
                shared = sorted(set(rates[a]) & set(rates[b]))
                n = len(shared)
                coverage[a][b] = n
                value = 1.0 if a == b else self.surface_prior
                if a != b and n >= self.min_shared_players:
                    x, y = ([rates[a][p] for p in shared], [rates[b][p] for p in shared])
                    mx, my = (sum(x) / n, sum(y) / n)
                    xx, yy = (sum(((v - mx) ** 2 for v in x)), sum(((v - my) ** 2 for v in y)))
                    if xx > 1e-15 and yy > 1e-15:
                        corr = sum(((u - mx) * (v - my) for u, v in zip(x, y))) / math.sqrt(xx * yy)
                        strength = n / (n + self.surface_shrink)
                        value = max(0.0, min(1.0, strength * corr + (1 - strength) * self.surface_prior))
                matrix[a][b] = value
        state = (matrix, coverage, cutoff)
        self._surface_cache[cutoff] = state
        return state

    def surface_matrix(self, asof):
        now = _sf_stamp(asof)
        if now is None:
            raise ValueError('asof must be a valid date or timestamp')
        matrix, _, _ = self._surface_state(now)
        return {key: dict(value) for key, value in matrix.items()}

    def surface_diagnostics(self, asof):
        now = _sf_stamp(asof)
        if now is None:
            raise ValueError('asof must be a valid date or timestamp')
        _, coverage, cutoff = self._surface_state(now)
        return {'learned_before': datetime.fromtimestamp(cutoff, timezone.utc).isoformat(), 'shared_players': {key: dict(value) for key, value in coverage.items()}}

    def _entries(self, player, target, now):
        matrix, _, _ = self._surface_state(now)
        result = []
        for r in self._past(player, now):
            sw = 1.0 if r['surface'] == target else matrix.get(target, {}).get(r['surface'], self.surface_prior)
            weight = _sf_temporal_weight((now - r['date']) / _SF_YEAR, self.discount_factor) * sw
            if weight > 0:
                result.append((r, weight))
        return result

    def _pool(self, entries):
        totals = {key: [0.0, 0.0] for key in _SF_PRIOR_RATES}
        for r, weight in entries:
            for key, pair in r['counts'].items():
                if pair is not None:
                    totals[key][0] += weight * pair[0]
                    totals[key][1] += weight * pair[1]
        return totals

    def _profile(self, entries, now):
        result = {}
        for key, (n, d) in self._pool(entries).items():
            result[key] = (n + self.prior_points * _SF_PRIOR_RATES[key]) / (d + self.prior_points)
            result[key + '_opportunities'] = d
        mass = sum((w for _, w in entries))
        squares = sum((w * w for _, w in entries))
        result['win_rate'] = (sum((w * r['win'] for r, w in entries)) + 0.5 * self.prior_matches) / (mass + self.prior_matches)
        result['wsp'] = result['first_in'] * result['first_win'] + (1 - result['first_in']) * result['second_win']
        result['complete'] = result['wsp'] * result['return_won']
        result['serve_return_product'] = result['serve_won'] * result['return_won']
        result['history_matches'] = float(len(entries))
        result['effective_matches'] = mass * mass / squares if squares else 0.0
        valid = [(r, w) for r, w in entries if any((p is not None for p in r['counts'].values()))]
        result['stat_coverage'] = len(valid) / len(entries) if entries else 0.0
        result['latest_age_days'] = min(((now - r['date']) / _SF_DAY for r, _ in entries)) if entries else self.history_days
        for key in _SF_RESIDUALS:
            observed = [(r['residuals'][key], w) for r, w in entries if r['residuals'][key] is not None]
            result[key] = sum((v * w for v, w in observed)) / (sum((w for _, w in observed)) + self.prior_matches)
            result[key + '_matches'] = float(len(observed))
        return result

    def snapshot(self, player_id, surface, asof):
        player, target, now = (_sf_identity(player_id), _sf_surface(surface), _sf_stamp(asof))
        if now is None:
            raise ValueError('asof must be a valid date or timestamp')
        if self._profile_time != now:
            self._profile_cache.clear()
            self._profile_time = now
        key = (player, target)
        if key not in self._profile_cache:
            entries = self._entries(player, target, now)
            result = self._profile(entries, now)
            for prefix, subset in (('surface_', [(r, w) for r, w in entries if r['surface'] == target]), ('recent_', [(r, w) for r, w in entries if now - r['date'] <= 90 * _SF_DAY])):
                p = self._profile(subset, now)
                for name in ('serve_won', 'return_won', 'first_win', 'second_win', 'win_rate', 'history_matches'):
                    result[prefix + name] = p[name]
            self._profile_cache[key] = result
        return dict(self._profile_cache[key])

    def pair_features(self, player_a, player_b, surface, asof):
        """Signed A-minus-B features plus explicitly invariant coverage.

        Common opponents are averaged equally; per-opponent count rates
        shrink toward the same fixed prior before subtraction, and the final
        contrast shrinks by n/(n+prior_opponents). A comparison is included
        only when BOTH players have evidence for that metric against C.
        """
        a, b, target, now = (_sf_identity(player_a), _sf_identity(player_b), _sf_surface(surface), _sf_stamp(asof))
        if a == b or now is None:
            raise ValueError('Require distinct players and a valid asof')
        pa, pb = (self.snapshot(a, target, asof), self.snapshot(b, target, asof))
        result = {}
        for name in pa:
            if name.endswith('_opportunities'):
                result['diff_' + name] = math.log1p(pa[name]) - math.log1p(pb[name])
                result['coverage_' + name] = math.log1p(min(pa[name], pb[name]))
            else:
                result['diff_' + name] = pa[name] - pb[name]
                if name.endswith('_matches') or name == 'stat_coverage':
                    result['coverage_' + name] = min(pa[name], pb[name])
        result['diff_serveadv'] = pa['wsp'] - pb['return_won'] - (pb['wsp'] - pa['return_won'])
        histories = [self._entries(p, target, now) for p in (a, b)]
        groups = []
        for entries in histories:
            group = defaultdict(list)
            for r, w in entries:
                group[r['opponent']].append((r, w))
            groups.append(group)
        common = sorted(set(groups[0]) & set(groups[1]))
        pools = {c: (self._pool(groups[0][c]), self._pool(groups[1][c])) for c in common}
        result['coverage_common_opponents'] = float(len(common))
        for metric in ('serve_won', 'return_won', 'first_win', 'second_win', 'return1', 'return2'):
            contrasts = []
            for left, right in pools.values():
                n1, d1 = left[metric]
                n2, d2 = right[metric]
                if d1 > 0 and d2 > 0:
                    prior = self.prior_points * _SF_PRIOR_RATES[metric]
                    contrasts.append((n1 + prior) / (d1 + self.prior_points) - (n2 + prior) / (d2 + self.prior_points))
            n = len(contrasts)
            result['diff_common_' + metric] = sum(contrasts) / (n + self.prior_opponents)
            result['coverage_common_' + metric] = float(n)
        contrasts = []
        for c in common:
            rates = []
            for group in groups:
                entries = group[c]
                mass = sum((w for _, w in entries))
                rates.append((sum((w * r['win'] for r, w in entries)) + 0.5 * self.prior_matches) / (mass + self.prior_matches))
            contrasts.append(rates[0] - rates[1])
        result['diff_common_win_rate'] = sum(contrasts) / (len(contrasts) + self.prior_opponents)
        return result


# ==================== momentum_features.py ====================
"""Past-only form and optional charted-shot features for CourtIQ.

Lei et al., Rhythms of Victory (2024), uses within-match point momentum.
Here the idea is ADAPTED to decayed surprises from completed matches; this
is not their point recurrence, psychological measurement, or 83.52% result.
Residuals must be computed against pre-update expectations by the caller.

Charted match dates do not establish when observations were published.
Charting defaults OFF. An explicit availability CSV (match_id,published_at)
or source_available_at enables timestamp-verified records. research_enabled
allows unknown-publication records, prominently identified in provenance.
Unknown records become eligible after the following UTC day, never on the
charted match day; that conservative lag still does not prove availability.
"""
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, date, timezone
from pathlib import Path
import math
import unicodedata
_MF_DAY = 86400.0

def _mf_num(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None

def _mf_stamp(value):
    try:
        if isinstance(value, datetime):
            d = value
        elif isinstance(value, date):
            d = datetime.combine(value, datetime.min.time())
        else:
            s = str(value).strip()
            if s.endswith('.0'):
                s = s[:-2]
            d = datetime.strptime(s, '%Y%m%d') if len(s) == 8 and s.isdigit() else datetime.fromisoformat(s.replace('Z', '+00:00'))
        return _mf_num((d if d.tzinfo else d.replace(tzinfo=timezone.utc)).timestamp())
    except (ValueError, TypeError, OverflowError):
        return None

def _mf_key(value):
    s = unicodedata.normalize('NFKD', str(value))
    return ' '.join(''.join((c for c in s if not unicodedata.combining(c))).replace('_', ' ').casefold().split())

def _mf_positive(name, value):
    number = _mf_num(value)
    if number is None or number <= 0:
        raise ValueError(name + ' must be finite and positive')
    return number

class MomentumFeatures:
    """update(player_id, record); profile(player_id, surface, asof).

    Records require date, known_at, surface, completed=True. Optional
    win_residual, serve_residual, return_residual must lie in [-1,1]. A
    match_id enables duplicate protection. Updates may arrive out of order;
    profiles still use only date <= known_at < asof. Prior matches shrink
    residuals toward zero; they are an assumption, not measured confidence.
    """
    METRICS = ('win', 'serve', 'return')

    def __init__(self, half_life_days=90, long_half_life_days=365, window_days=730, other_surface_weight=0.25, prior_matches=5):
        self.short = _mf_positive('half_life_days', half_life_days)
        self.long = _mf_positive('long_half_life_days', long_half_life_days)
        self.window = _mf_positive('window_days', window_days)
        self.prior = _mf_positive('prior_matches', prior_matches)
        if _mf_num(other_surface_weight) is None or not 0 <= other_surface_weight <= 1:
            raise ValueError('other_surface_weight must be in [0,1]')
        self.surface_weight = float(other_surface_weight)
        self.records, self.times, self.seen = (defaultdict(list), defaultdict(list), {})

    def update(self, player_id, record):
        if record.get('completed') is not True:
            raise ValueError('Momentum accepts only completed matches')
        played, known = (_mf_stamp(record.get('date')), _mf_stamp(record.get('known_at')))
        if played is None or known is None or known < played:
            raise ValueError('Require valid date <= known_at')
        player = str(player_id)
        values = []
        for name in self.METRICS:
            value = _mf_num(record.get(name + '_residual'))
            if value is not None and (not -1 <= value <= 1):
                raise ValueError(name + '_residual must be in [-1,1]')
            values.append(value)
        entry = (played, known, _mf_key(record.get('surface', '')), tuple(values))
        if record.get('match_id') is not None:
            token = (player, str(record['match_id']))
            if token in self.seen:
                if self.seen[token] != entry:
                    raise ValueError('Conflicting update for an existing player/match_id')
                return False
            self.seen[token] = entry
        position = bisect_right(self.times[player], known)
        self.times[player].insert(position, known)
        self.records[player].insert(position, entry)
        return True

    def profile(self, player_id, surface, asof):
        now = _mf_stamp(asof)
        if now is None:
            raise ValueError('asof must be a valid timestamp')
        player, target = (str(player_id), _mf_key(surface))
        times = self.times.get(player, [])
        start, stop = (bisect_left(times, now - self.window * _MF_DAY), bisect_left(times, now))
        records = [r for r in self.records.get(player, [])[start:stop] if 0 <= now - r[0] <= self.window * _MF_DAY]
        result = {'momentum_history_matches': float(len(records))}
        for index, metric in enumerate(self.METRICS):
            valid = [r for r in records if r[3][index] is not None and (r[2] == target or self.surface_weight > 0)]
            for label, half in (('short', self.short), ('long', self.long)):
                weighted = [(2 ** (-(now - r[0]) / _MF_DAY / half) * (1.0 if r[2] == target else self.surface_weight), r[3][index]) for r in valid]
                den = sum((w for w, _ in weighted))
                result['momentum_' + metric + '_' + label] = sum((w * v for w, v in weighted)) / (den + self.prior)
            result['momentum_' + metric + '_trend'] = result['momentum_' + metric + '_short'] - result['momentum_' + metric + '_long']
            result['momentum_' + metric + '_matches'] = float(len(valid))
        return result
_MF_SHOT_SPECS = {'ShotTypes': {'Fside': (('forehand_winner', 'winners', 'shots'), ('forehand_error', 'unforced', 'shots'), ('forehand_forced', 'induced_forced', 'shots')), 'Bside': (('backhand_winner', 'winners', 'shots'), ('backhand_error', 'unforced', 'shots'), ('backhand_forced', 'induced_forced', 'shots'))}, 'ServeBasics': {'1': (('first_serve', 'pts_won', 'pts'),), '2': (('second_serve', 'pts_won', 'pts'),)}, 'ReturnOutcomes': {'v1st': (('return_first', 'pts_won', 'pts'),), 'v2nd': (('return_second', 'pts_won', 'pts'),)}, 'NetPoints': {'NetPoints': (('net_success', 'pts_won', 'net_pts'),)}}
_MF_SHOT_NAMES = tuple((x[0] for spec in _MF_SHOT_SPECS.values() for row in spec.values() for x in row))

class ChartedShotStore:
    """Optional CSV shot store. Alias keys may be source names or actual IDs.

    Numeric player=1/2 and labels player1/player2 refer to metadata sides.
    Other numeric values are actual player IDs; names resolve only through
    explicit aliases or remain names. No fuzzy identity matching is used.
    Availability CSV must have match_id and published_at. With research
    disabled, matches absent from that CSV are excluded. A global
    source_available_at is a conservative known availability of a snapshot.
    """

    def __init__(self, base_dir=None, research_enabled=False, availability_path=None, player_aliases=None, source_available_at=None, half_life_days=365, window_days=730, other_surface_weight=0.25):
        self.half = _mf_positive('half_life_days', half_life_days)
        self.window = _mf_positive('window_days', window_days)
        if _mf_num(other_surface_weight) is None or not 0 <= other_surface_weight <= 1:
            raise ValueError('other_surface_weight must be in [0,1]')
        self.surface_weight = float(other_surface_weight)
        self.research = bool(research_enabled)
        self.enabled = bool(research_enabled or availability_path or source_available_at)
        self.aliases = {_mf_key(k): str(v) for k, v in (player_aliases or {}).items()}
        self.records, self.times = (defaultdict(list), defaultdict(list))
        self.provenance = dict(enabled=self.enabled, research_enabled=self.research, unknown_publication_matches=0, excluded_unknown_availability=0, invalid_metadata=0, invalid_statistics=0, missing_files=[], loaded_player_matches=0)
        if self.enabled:
            if base_dir is None:
                raise ValueError('base_dir is required when charting is enabled')
            self._load(Path(base_dir), availability_path, source_available_at)

    @property
    def notes(self):
        if not self.enabled:
            return ['Charted-shot features are disabled by default.']
        result = ['Fside/Bside are forehand-side/backhand-side proxies, not player skill ratings.']
        if self.provenance['unknown_publication_matches']:
            result.append('Research charting uses match dates with a one-day lag where publication time is unknown; historical availability is unverified.')
        if self.provenance['excluded_unknown_availability']:
            result.append('Charted matches without verified availability timestamps were excluded.')
        if self.provenance['missing_files']:
            result.append('Missing optional charting files: ' + ', '.join(self.provenance['missing_files']))
        return result

    def _identity(self, raw, meta):
        label = _mf_key(raw)
        if label.endswith('.0') and label[:-2].isdigit():
            label = label[:-2]
        compact = label.replace(' ', '')
        if compact in ('1', 'p1', 'player1', '2', 'p2', 'player2'):
            side = 0 if compact.endswith('1') else 1
            label = _mf_key(meta['players'][side])
        return self.aliases.get(label, label)

    def _load(self, base, availability_path, source_available_at):
        import pandas as pd
        published = {}
        if availability_path:
            availability = pd.read_csv(availability_path, dtype=str)
            if not {'match_id', 'published_at'} <= set(availability.columns):
                raise ValueError('Availability CSV requires match_id,published_at')
            for row in availability.to_dict('records'):
                timestamp = _mf_stamp(row['published_at'])
                if timestamp is None:
                    raise ValueError('Invalid published_at for ' + str(row['match_id']))
                mid = str(row['match_id'])
                if mid in published and published[mid] != timestamp:
                    raise ValueError('Conflicting publication timestamps for ' + mid)
                published[mid] = timestamp
        global_known = _mf_stamp(source_available_at) if source_available_at is not None else None
        if source_available_at is not None and global_known is None:
            raise ValueError('Invalid source_available_at')
        metadata = pd.read_csv(base / 'charting-m-matches.csv', dtype=str)
        required = {'match_id', 'Date', 'Surface', 'Player 1', 'Player 2'}
        if not required <= set(metadata.columns):
            raise ValueError('Missing charting metadata columns: ' + str(sorted(required - set(metadata.columns))))
        matches = {}
        for row in metadata.to_dict('records'):
            mid, played = (str(row['match_id']), _mf_stamp(row['Date']))
            if played is None or any((pd.isna(row[k]) for k in ('Player 1', 'Player 2', 'Surface'))):
                self.provenance['invalid_metadata'] += 1
                continue
            known = published.get(mid, global_known)
            if known is None and 'published_at' in row and (not pd.isna(row['published_at'])):
                known = _mf_stamp(row['published_at'])
            unknown = known is None
            if unknown and (not self.research):
                self.provenance['excluded_unknown_availability'] += 1
                continue
            known = played + _MF_DAY if unknown else known
            if known < played:
                self.provenance['invalid_metadata'] += 1
                continue
            self.provenance['unknown_publication_matches'] += int(unknown)
            meta = dict(date=played, known_at=known, unknown=unknown, surface=_mf_key(row['Surface']), players=(row['Player 1'], row['Player 2']))
            if mid in matches and matches[mid] != meta:
                raise ValueError('Conflicting charted match metadata for ' + mid)
            matches[mid] = meta
        rows = {}
        for source, selectors in _MF_SHOT_SPECS.items():
            path = base / ('charting-m-stats-' + source + '.csv')
            if not path.exists():
                self.provenance['missing_files'].append(path.name)
                continue
            for frame in pd.read_csv(path, dtype=str, chunksize=50000):
                if not {'match_id', 'player', 'row'} <= set(frame.columns):
                    raise ValueError('Missing identifiers in ' + path.name)
                frame = frame[frame['row'].isin(selectors)]
                for row in frame.to_dict('records'):
                    mid = str(row['match_id'])
                    if mid not in matches or pd.isna(row['player']):
                        continue
                    if source == 'ShotTypes':
                        pieces = [_mf_num(row.get(k)) for k in ('winners', 'unforced', 'induced_forced', 'shots')]
                        if all((x is not None for x in pieces)) and sum(pieces[:3]) > pieces[3]:
                            self.provenance['invalid_statistics'] += 1
                            continue
                    meta = matches[mid]
                    player = self._identity(row['player'], meta)
                    entry = rows.setdefault((mid, player), dict(meta, pairs={}, invalid=set()))
                    for name, num_col, den_col in selectors[row['row']]:
                        numerator, denominator = (_mf_num(row.get(num_col)), _mf_num(row.get(den_col)))
                        if numerator is None or denominator is None or (not 0 <= numerator <= denominator) or (denominator <= 0):
                            self.provenance['invalid_statistics'] += 1
                            continue
                        pair = (numerator, denominator)
                        if name in entry['invalid']:
                            continue
                        if name in entry['pairs'] and entry['pairs'][name] != pair:
                            del entry['pairs'][name]
                            entry['invalid'].add(name)
                            self.provenance['invalid_statistics'] += 1
                        else:
                            entry['pairs'][name] = pair
        for (_, player), entry in rows.items():
            if entry['pairs']:
                self.records[player].append(entry)
        for player, entries in self.records.items():
            entries.sort(key=lambda r: r['known_at'])
            self.times[player] = [r['known_at'] for r in entries]
        self.provenance['loaded_player_matches'] = sum(map(len, self.records.values()))

    def profile(self, player_id, surface, asof):
        now = _mf_stamp(asof)
        if now is None:
            raise ValueError('asof must be a valid timestamp')
        key = self.aliases.get(_mf_key(player_id), _mf_key(player_id))
        times = self.times.get(key, [])
        start, stop = (bisect_left(times, now - self.window * _MF_DAY), bisect_left(times, now))
        entries = [r for r in self.records.get(key, [])[start:stop] if 0 < now - r['date'] <= self.window * _MF_DAY]
        totals = {name: [0.0, 0.0, 0] for name in _MF_SHOT_NAMES}
        ages, same_matches, unknown = ([], 0, 0)
        for r in entries:
            same = r['surface'] == _mf_key(surface)
            age = (now - r['date']) / _MF_DAY
            weight = 2 ** (-age / self.half) * (1.0 if same else self.surface_weight)
            if weight <= 0:
                continue
            ages.append(age)
            same_matches += int(same)
            unknown += int(r['unknown'])
            for name, (n, d) in r['pairs'].items():
                totals[name][0] += weight * n
                totals[name][1] += weight * d
                totals[name][2] += 1
        result = dict(shot_enabled=float(self.enabled), shot_history_matches=float(len(ages)), shot_same_surface_matches=float(same_matches), shot_unknown_publication_matches=float(unknown), shot_latest_match_age_days=min(ages) if ages else math.nan)
        for name, (n, d, count) in totals.items():
            result['shot_' + name + '_rate'] = n / d if d else math.nan
            result['shot_' + name + '_log_opportunities'] = math.log1p(d)
            result['shot_' + name + '_matches'] = float(count)
        return result


# ==================== paper_estimator.py ====================
"""Chronologically selected, serializable prematch estimator for CourtIQ.

The logistic model's signed-difference/context interactions are inspired by
Sipko's interaction modeling. They are not a replication of his exact model.
Rhythms' point-level momentum scores are not used as prematch observations.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss

def _pe_time(value):
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError('A valid chronological cutoff is required.')
    return parsed.tz_localize('UTC').tz_localize(None) if parsed.tzinfo is None else parsed.tz_convert('UTC').tz_localize(None)

def _pe_frame(value, columns=None):
    frame = value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate feature names are not supported.')
    if columns is not None:
        if set(frame.columns) != set(columns):
            raise ValueError('Prediction features do not match the fitted feature schema.')
        frame = frame.loc[:, columns]
    frame = frame.astype(float)
    if np.isinf(frame.to_numpy()).any():
        raise ValueError('Infinite feature values must be fixed before fitting or inference.')
    return frame.reset_index(drop=True)

def _pe_weights(dates, anchor, half_life):
    if half_life is None:
        return np.ones(len(dates), dtype=float)
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError('Sample-weight half-life must be positive.')
    ages = (anchor - dates).total_seconds().to_numpy() / (365.2425 * 86400)
    weights = np.exp2(-ages / float(half_life))
    if not np.isfinite(weights).all() or weights.mean() <= 0:
        raise ValueError('Invalid time-decay weights.')
    return weights / weights.mean()

def _pe_interactions(X, reverse):
    """Select bounded, outcome-independent interactions from training rows only."""
    signed, contexts = ([], [])
    for name in X:
        a, b = (X[name].to_numpy(), reverse[name].to_numpy())
        finite = np.isfinite(a) & np.isfinite(b)
        if not finite.any() or not np.array_equal(np.isnan(a), np.isnan(b)):
            continue
        if np.allclose(a[finite], -b[finite], rtol=1e-10, atol=1e-10) and np.any(a[finite] != 0):
            signed.append(name)
        elif np.allclose(a[finite], b[finite], rtol=1e-10, atol=1e-10):
            lowered = str(name).lower()
            if any((token in lowered for token in ('surface', 'best_of', 'format'))):
                if 1 < len(np.unique(a[finite])) <= 12:
                    contexts.append(name)
    priorities = ('serve', 'return', 'elo', 'common', 'trend', 'form', 'complete', 'momentum')
    signed.sort(key=lambda name: (next((i for i, word in enumerate(priorities) if word in str(name).lower()), 99), str(name)))
    return [(name, context) for name in signed[:20] for context in contexts[:8]]

def _pe_matrix(X, interactions):
    main = X.to_numpy(dtype=float)
    if not interactions:
        return main
    extra = np.column_stack([(X[a] * X[b]).to_numpy() for a, b in interactions])
    matrix = np.column_stack([main, extra])
    if np.isinf(matrix).any():
        raise ValueError('An interaction overflowed; check feature magnitudes.')
    return matrix

def _pe_fit_logistic(X, reverse, y, weights, interactions):
    combined = pd.concat([X, reverse], ignore_index=True)
    target = np.concatenate([y, 1 - y])
    sample_weight = np.concatenate([weights, weights]) / 2
    imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
    values = imputer.fit_transform(_pe_matrix(combined, interactions))
    keep = np.ptp(values, axis=0) > 0
    if not keep.any():
        raise ValueError('No varying logistic features remain after training-only imputation.')
    scaler = StandardScaler().fit(values[:, keep])
    values = scaler.transform(values[:, keep])
    if not np.isfinite(values).all():
        raise ValueError('Logistic preprocessing produced nonfinite values.')
    classifier = LogisticRegression(C=1.0, solver='liblinear', max_iter=1000, random_state=42)
    classifier.fit(values, target, sample_weight=sample_weight)
    if not np.isfinite(classifier.coef_).all() or not np.isfinite(classifier.intercept_).all():
        raise RuntimeError('Logistic regression produced nonfinite parameters.')
    if np.max(classifier.n_iter_) >= classifier.max_iter:
        raise RuntimeError('Logistic regression did not converge within its iteration limit.')
    return dict(imputer=imputer, scaler=scaler, classifier=classifier, interactions=interactions, keep_columns=keep, dropped_constant_columns=int((~keep).sum()))

def _pe_logistic_probability(model, X):
    values = model['imputer'].transform(_pe_matrix(X, model['interactions']))
    values = model['scaler'].transform(values[:, model['keep_columns']])
    if not np.isfinite(values).all():
        raise ValueError('Logistic inference received nonfinite processed values.')
    classifier = model['classifier']
    score = np.einsum('ij,j->i', values, classifier.coef_[0], optimize=False) + classifier.intercept_[0]
    return 1.0 / (1.0 + np.exp(-np.clip(score, -700, 700)))

def _pe_fit_catboost(X, reverse, y, weights, iterations=500, threads=4):
    from catboost import CatBoostClassifier
    classifier = CatBoostClassifier(iterations=iterations, depth=5, learning_rate=0.035, l2_leaf_reg=5, loss_function='Logloss', verbose=False, thread_count=threads, random_seed=42, allow_writing_files=False)
    classifier.fit(pd.concat([X, reverse], ignore_index=True), np.concatenate([y, 1 - y]), sample_weight=np.concatenate([weights, weights]) / 2)
    return classifier

def _pe_symmetric(model, kind, X, reverse):
    probability = _pe_logistic_probability if kind == 'logistic' else lambda fitted, values: fitted.predict_proba(values)[:, 1]
    p = (probability(model, X) + 1 - probability(model, reverse)) / 2
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise RuntimeError('The estimator returned invalid probabilities.')
    return p

def _pe_metrics(y, p):
    return dict(accuracy=float(accuracy_score(y, p >= 0.5)), log_loss=float(log_loss(y, p, labels=[0, 1])), auc=float(roc_auc_score(y, p)), brier=float(brier_score_loss(y, p)))

def estimator_fit(X, Xswap, y, dates, available_at, train_end, selection_year=2024, start=2010, half_life=5, iterations=500, threads=4):
    """Select internally, refit by train_end, return a joblib-serializable bundle.

    2025/2026 are excluded when selection_year=2024. Both event time and label
    availability must precede a training cutoff. The five fixed mixture weights
    are selected by 2024 accuracy, with log loss as the tie breaker.
    """
    X = _pe_frame(X)
    reverse = _pe_frame(Xswap, list(X.columns))
    y = np.asarray(y)
    dates = pd.DatetimeIndex(pd.to_datetime(dates, utc=True)).tz_localize(None)
    available = pd.DatetimeIndex(pd.to_datetime(available_at, utc=True)).tz_localize(None)
    if not len(X) == len(reverse) == len(y) == len(dates) == len(available):
        raise ValueError('Feature rows, labels, and timestamps must align.')
    if len(X) == 0 or y.ndim != 1 or (not np.isin(y, [0, 1]).all()) or dates.hasnans or available.hasnans:
        raise ValueError('Expected nonempty binary labels and valid timestamps.')
    if np.any(available < dates):
        raise ValueError('A label cannot be available before its forecast time.')
    if not isinstance(iterations, int) or not isinstance(threads, int) or iterations <= 0 or (threads <= 0):
        raise ValueError('iterations and threads must be positive integers.')
    end = _pe_time(train_end)
    selection_start = pd.Timestamp(year=int(selection_year), month=1, day=1)
    selection_end = pd.Timestamp(year=int(selection_year) + 1, month=1, day=1)
    if not selection_start < end <= selection_end:
        raise ValueError('train_end must follow selection-year start and not enter later reserved years.')
    beginning = pd.Timestamp(year=int(start), month=1, day=1)
    inner = (dates >= beginning) & (dates < selection_start) & (available < selection_start)
    validation = (dates >= selection_start) & (dates < end) & (available < end)
    final = (dates >= beginning) & (dates < end) & (available < end)
    for name, mask in (('inner training', inner), ('inner validation', validation)):
        if not mask.any() or len(np.unique(y[mask])) != 2:
            raise ValueError(name + ' needs eligible examples of both outcomes.')
    interactions = _pe_interactions(X.loc[inner], reverse.loc[inner])
    weights = _pe_weights(dates[inner], selection_start, half_life)
    print('Fitting logistic interaction and CatBoost models on inner history...', flush=True)
    logistic = _pe_fit_logistic(X.loc[inner], reverse.loc[inner], y[inner], weights, interactions)
    catboost = _pe_fit_catboost(X.loc[inner], reverse.loc[inner], y[inner], weights, iterations, threads)
    p_lr = _pe_symmetric(logistic, 'logistic', X.loc[validation], reverse.loc[validation])
    p_cb = _pe_symmetric(catboost, 'catboost', X.loc[validation], reverse.loc[validation])
    trials = []
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        trials.append(dict(catboost_weight=weight, **_pe_metrics(y[validation], (1 - weight) * p_lr + weight * p_cb)))
    choice = min(trials, key=lambda row: (-row['accuracy'], row['log_loss'], row['catboost_weight']))
    weight = choice['catboost_weight']
    print('Selected CatBoost weight %.2f on %d inner validation matches; refitting eligible history...' % (weight, validation.sum()), flush=True)
    final_weights = _pe_weights(dates[final], end, half_life)
    models = {}
    if weight < 1:
        models['logistic'] = _pe_fit_logistic(X.loc[final], reverse.loc[final], y[final], final_weights, interactions)
    if weight > 0:
        models['catboost'] = _pe_fit_catboost(X.loc[final], reverse.loc[final], y[final], final_weights, iterations, threads)
    audit = dict(selection_year=int(selection_year), training_start=int(start), train_end_exclusive=end.isoformat(), half_life_years=half_life, inner_training_rows=int(inner.sum()), inner_validation_rows=int(validation.sum()), final_training_rows=int(final.sum()), latest_training_forecast=dates[final].max().isoformat(), latest_training_label_availability=available[final].max().isoformat(), interactions=[list(p) for p in interactions], inner_validation_trials=trials, selected=choice, selection_metric='accuracy; log loss breaks ties', validation_note='Inner-selection metrics are not independent final-test results.', calibration='No separate probability calibration fitted.')
    return dict(schema_version=1, kind='courtiq_paired_prematch', features=list(X.columns), models=models, catboost_weight=weight, audit=audit, interaction_pairs=interactions, training_config=dict(iterations=iterations, threads=threads, start=start, half_life=half_life))

def refit_selected(bundle, X, Xswap, y, dates, available_at, train_end, start=2010, half_life=5):
    """Refit a frozen choice for deployment; never select or report test scores."""
    if bundle.get('kind') != 'courtiq_paired_prematch' or bundle.get('schema_version') != 1:
        raise ValueError('Unsupported CourtIQ estimator bundle.')
    X, reverse = (_pe_frame(X, bundle['features']), _pe_frame(Xswap, bundle['features']))
    y = np.asarray(y)
    dates = pd.DatetimeIndex(pd.to_datetime(dates, utc=True)).tz_localize(None)
    available = pd.DatetimeIndex(pd.to_datetime(available_at, utc=True)).tz_localize(None)
    if not len(X) == len(reverse) == len(y) == len(dates) == len(available):
        raise ValueError('Refit features, labels, and timestamps must align.')
    if y.ndim != 1 or not np.isin(y, [0, 1]).all() or dates.hasnans or available.hasnans or np.any(available < dates):
        raise ValueError('Refit requires valid labels and label-availability timestamps.')
    end = _pe_time(train_end)
    eligible = (dates >= pd.Timestamp(year=int(start), month=1, day=1)) & (dates < end) & (available < end)
    if not eligible.any() or len(np.unique(y[eligible])) != 2:
        raise ValueError('Refit needs eligible examples of both outcomes.')
    weights = _pe_weights(dates[eligible], end, half_life)
    weight = bundle['catboost_weight']
    interactions = bundle['interaction_pairs']
    config = dict(bundle['training_config'], start=start, half_life=half_life)
    models = {}
    if weight < 1:
        models['logistic'] = _pe_fit_logistic(X.loc[eligible], reverse.loc[eligible], y[eligible], weights, interactions)
    if weight > 0:
        models['catboost'] = _pe_fit_catboost(X.loc[eligible], reverse.loc[eligible], y[eligible], weights, config['iterations'], config['threads'])
    updated = dict(bundle, models=models, training_config=config)
    updated['refit'] = dict(train_end_exclusive=end.isoformat(), training_rows=int(eligible.sum()), latest_training_forecast=dates[eligible].max().isoformat(), latest_label_availability=available[eligible].max().isoformat(), selection_changed=False, evaluation_note='Deployment refit only; no new validation or test performance was measured.')
    return updated

def estimator_predict(bundle, X, Xswap):
    """Predict P(player A wins), enforcing P(A,B) = 1 - P(B,A)."""
    if bundle.get('kind') != 'courtiq_paired_prematch' or bundle.get('schema_version') != 1:
        raise ValueError('Unsupported CourtIQ estimator bundle.')
    X = _pe_frame(X, bundle['features'])
    reverse = _pe_frame(Xswap, bundle['features'])
    if len(X) != len(reverse):
        raise ValueError('Both player orientations must have the same number of rows.')
    if not len(X):
        return np.array([], dtype=float)
    weight = bundle['catboost_weight']
    p = np.zeros(len(X), dtype=float)
    if weight < 1:
        p += (1 - weight) * _pe_symmetric(bundle['models']['logistic'], 'logistic', X, reverse)
    if weight > 0:
        p += weight * _pe_symmetric(bundle['models']['catboost'], 'catboost', X, reverse)
    return p


# ==================== research_pipeline.py ====================
"""Integrated paper-informed tennis training and named-player prediction.

Development module; distributed with its three helpers in one Python file.
Only historical matches update player states. ATP tournament start + 21 days
is an explicit availability proxy, not a verified match completion timestamp.
"""
import argparse
from collections import defaultdict
import copy
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import time
import unicodedata
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
RESEARCH_VERSION = '1.0'
RESEARCH_LIMITATIONS = ['ATP tourney_date is a tournament-start date. Results enter history only after a 21-day availability proxy.', 'Momentum is an adaptation to completed matches; it does not reproduce the point-winner task in Rhythms of Victory.', 'Feature rates and coverage describe recorded historical performance, not scouting scores or measured psychological states.', 'Previously inspected evaluation seasons are historical comparisons, not newly independent holdouts.', 'Evaluation scores belong to the evaluated model. A later deployment refit has no new independent performance score.']

def tennis_number(value, default=float('nan')):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (ValueError, TypeError):
        return default

def tennis_name(value):
    value = unicodedata.normalize('NFKD', str(value))
    value = ''.join((c for c in value if not unicodedata.combining(c)))
    return ' '.join(value.casefold().split())

def tennis_player_id(value, name):
    number = tennis_number(value)
    if math.isfinite(number) and number == int(number):
        return str(int(number))
    if pd.notna(value) and str(value).strip():
        return str(value).strip()
    return 'name:' + tennis_name(name)

def tennis_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def tennis_date(value):
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError('A valid forecast date is required.')
    if result.tzinfo is not None:
        result = result.tz_convert('UTC').tz_localize(None)
    return result.normalize()

def tennis_input_files(data_dir, updates_dir):
    files = {p.name: p for p in Path(data_dir).glob('atp_matches_*.csv') if re.fullmatch('atp_matches_\\d{4}\\.csv', p.name)}
    for p in Path(updates_dir).glob('atp_matches_*.csv'):
        if re.fullmatch('atp_matches_\\d{4}\\.csv', p.name) and int(p.stem[-4:]) >= 2025:
            files[p.name] = p
    if not files:
        raise ValueError('No ATP season CSVs found. Run from the project root or provide --data-dir.')
    return [files[k] for k in sorted(files)]

def load_tennis_matches(data_dir, updates_dir, delay_days=21):
    files = tennis_input_files(data_dir, updates_dir)
    frames = [pd.read_csv(p, low_memory=False) for p in files]
    raw = pd.concat(frames, ignore_index=True)
    required = ['tourney_date', 'winner_name', 'loser_name', 'surface', 'tourney_id', 'match_num']
    missing = [c for c in required if c not in raw]
    if missing:
        raise ValueError('ATP data is missing columns: ' + ', '.join(missing))
    raw['date'] = pd.to_datetime(raw.tourney_date.astype(str).str.replace('\\.0$', '', regex=True), format='%Y%m%d', errors='coerce')
    raw = raw.dropna(subset=['date', 'winner_name', 'loser_name', 'surface']).copy()
    raw = raw[raw.date >= pd.Timestamp('1990-01-01')].copy()
    for side in ('winner', 'loser'):
        source_ids = raw.get(side + '_id', pd.Series(np.nan, index=raw.index))
        raw[side + '_id'] = [tennis_player_id(i, n) for i, n in zip(source_ids, raw[side + '_name'])]
    raw['surface'] = raw.surface.astype(str).str.strip().str.title()
    raw['known_at'] = raw.date + pd.Timedelta(days=int(delay_days))
    score = raw.get('score', pd.Series('', index=raw.index)).fillna('').astype(str)
    raw['completed'] = ~score.str.contains('W/O|WALKOVER|RET|DEF|ABD', case=False, regex=True)
    raw['retirement_proxy'] = score.str.contains('RET', case=False, regex=True)
    raw = raw[raw.winner_id != raw.loser_id].copy()
    raw['event_key'] = raw.get('tourney_name', pd.Series('', index=raw.index)).map(tennis_name)
    raw['match_id'] = [str(t) + ':' + str(m) + ':' + ':'.join(sorted((w, l))) for t, m, w, l in zip(raw.tourney_id, raw.match_num, raw.winner_id, raw.loser_id)]
    if (raw.groupby('match_id').winner_id.nunique() > 1).any():
        raise ValueError('Conflicting winners for the same match ID; reconcile the input files.')
    raw = raw.drop_duplicates('match_id', keep='last')
    raw = raw.sort_values(['date', 'tourney_id', 'match_num', 'match_id'], kind='mergesort').reset_index(drop=True)
    return (raw, {str(p): tennis_sha(p) for p in files})

def tennis_aliases(frame):
    lookup = defaultdict(set)
    names = {}
    for side in ('winner', 'loser'):
        for player, name in zip(frame[side + '_id'], frame[side + '_name']):
            lookup[tennis_name(name)].add(str(player))
            names[str(player)] = str(name)
    aliases = {key: next(iter(ids)) for key, ids in lookup.items() if len(ids) == 1}
    aliases.update({key: key for key in names})
    return (aliases, names)

def tennis_safe_rate(numerator, denominator):
    n, d = (tennis_number(numerator), tennis_number(denominator))
    return n / d if math.isfinite(n) and math.isfinite(d) and (d > 0) and (0 <= n <= d) else float('nan')

class TennisFeatureEngine:
    """One historical state and feature function shared by training and serving."""

    def __init__(self, config=None, charts=None):
        self.config = dict(config or {})
        self.sipko = SipkoFeatures(discount_factor=self.config.get('history_discount', 0.8), history_days=self.config.get('history_days', 1826), prior_points=self.config.get('prior_points', 50.0))
        self.momentum = MomentumFeatures()
        self.charts = charts or ChartedShotStore()
        self.elo = defaultdict(lambda: 1500.0)
        self.surface_elo = defaultdict(lambda: 1500.0)
        self.history = defaultdict(list)
        self.metadata = {}
        self.pending = []
        self.sequence = 0
        self.current_date = None
        self.latest_known_tournament = None

    def export_state(self):
        """Plain-data snapshot: no locally defined classes or lambda factories."""
        sipko = dict(vars(self.sipko))
        sipko.update(_profile_cache={}, _profile_time=None, _surface_cache={})
        return {'sipko': sipko, 'momentum': dict(vars(self.momentum)), 'elo': dict(self.elo), 'surface_elo': dict(self.surface_elo), 'history': dict(self.history), 'metadata': dict(self.metadata), 'pending': list(self.pending), 'sequence': self.sequence, 'current_date': self.current_date, 'latest_known_tournament': self.latest_known_tournament}

    @classmethod
    def from_state(cls, config, charts, state):
        engine = cls(config, charts)
        state = copy.deepcopy(state)
        vars(engine.sipko).update(state['sipko'])
        vars(engine.momentum).update(state['momentum'])
        engine.elo.update(state['elo'])
        engine.surface_elo.update(state['surface_elo'])
        engine.history.update(state['history'])
        for name in ('metadata', 'pending', 'sequence', 'current_date', 'latest_known_tournament'):
            setattr(engine, name, state[name])
        return engine

    def advance(self, asof):
        asof = tennis_date(asof)
        if self.current_date is not None and asof < self.current_date:
            raise ValueError('Feature state cannot move backwards; rebuild state for an earlier date.')
        while self.pending and self.pending[0][0] < asof:
            _, _, row, records = heapq.heappop(self.pending)
            self._release(row, records)
        self.current_date = asof

    def _release(self, row, records):
        w, l, surface = (row['winner_id'], row['loser_id'], row['surface'])
        if row['completed']:
            for ratings, a, b in ((self.elo, w, l), (self.surface_elo, (w, surface), (l, surface))):
                expected = 1 / (1 + 10 ** ((ratings[b] - ratings[a]) / 400))
                delta = 24 * (1 - expected)
                ratings[a] += delta
                ratings[b] -= delta
        for player, record in records:
            self.history[player].append(record)
            self.metadata[player] = record['player_metadata']
            if row['completed']:
                self.sipko.update(player, record)
                self.momentum.update(player, record)
        last = row['date']
        if self.latest_known_tournament is None or last > self.latest_known_tournament:
            self.latest_known_tournament = last

    def profiles(self, a, b, surface, asof):
        return (self.sipko.snapshot(a, surface, asof), self.sipko.snapshot(b, surface, asof))

    def schedule(self, row, profiles=None):
        w, l, surface, date = (row['winner_id'], row['loser_id'], row['surface'], row['date'])
        if self.current_date != date:
            raise ValueError('Advance state to the match date before scheduling its result.')
        if profiles is None:
            profiles = self.profiles(w, l, surface, date)
        pw, pl = profiles
        expected = 1 / (1 + 10 ** ((self.elo[l] - self.elo[w]) / 400))
        records = []
        for player, side, own, opp, win, opponent, opponent_profile in ((w, 'winner', 'w_', 'l_', 1.0, l, pl), (l, 'loser', 'l_', 'w_', 0.0, w, pw)):
            rec = {'date': date, 'known_at': row['known_at'], 'surface': surface, 'opponent': opponent, 'win': win, 'completed': bool(row['completed']), 'match_id': row['match_id'], 'event_key': row['event_key'], 'retirement_proxy': bool(row['retirement_proxy']), 'minutes': tennis_number(row.get('minutes')), 'win_residual': win - (expected if win else 1 - expected)}
            own_map = {'svpt': 'svpt', 'first_in_count': '1stIn', 'first_won_count': '1stWon', 'second_won_count': '2ndWon', 'aces': 'ace', 'double_faults': 'df', 'bp_saved': 'bpSaved', 'bp_faced': 'bpFaced'}
            opp_map = {'return_svpt': 'svpt', 'return_first_in_count': '1stIn', 'return_first_won_count': '1stWon', 'return_second_won_count': '2ndWon', 'opp_bp_saved': 'bpSaved', 'opp_bp_faced': 'bpFaced'}
            for dest, source in own_map.items():
                rec[dest] = tennis_number(row.get(own + source))
            for dest, source in opp_map.items():
                rec[dest] = tennis_number(row.get(opp + source))
            valid_rates = observed_rates(rec)
            serve_won, return_won = (valid_rates['serve_won'], valid_rates['return_won'])
            rec['adj_serve'] = rec['serve_residual'] = serve_won - (1 - opponent_profile['return_won'])
            rec['adj_return'] = rec['return_residual'] = return_won - (1 - opponent_profile['serve_won'])
            rec['player_metadata'] = {field: tennis_number(row.get(side + '_' + field)) for field in ('rank', 'rank_points', 'age', 'ht')}
            rec['player_metadata'].update(date=date, known_at=row['known_at'], name=str(row[side + '_name']))
            records.append((player, rec))
        self.sequence += 1
        heapq.heappush(self.pending, (row['known_at'], self.sequence, row, records))

    def _player_context(self, player, surface, event, asof, supplied=None):
        h = [r for r in self.history[player] if r['known_at'] < asof]
        completed = [r for r in h if r['completed']]
        latest = self.metadata.get(player, {})
        supplied = supplied or {}
        out = {'elo': self.elo[player], 'surface_elo': self.surface_elo[player, surface]}
        for name in ('rank', 'rank_points', 'age', 'ht'):
            direct = tennis_number(supplied.get(name))
            value = direct if math.isfinite(direct) else tennis_number(latest.get(name))
            if name == 'age' and (not math.isfinite(direct)) and math.isfinite(value) and ('date' in latest):
                value += (asof - latest['date']).days / 365.2425
            if name in ('rank', 'rank_points'):
                out[name] = math.log1p(value) if math.isfinite(value) and value >= 0 else float('nan')
                out[name + '_provided'] = float(math.isfinite(direct))
            else:
                out[name] = value
        for n in (5, 10, 20, 50):
            past = completed[-n:]
            out['recent_win_%s' % n] = (sum((r['win'] for r in past)) + 1) / (len(past) + 2)
        event_history = [r for r in completed if r['event_key'] == tennis_name(event)]
        out['event_win'] = (sum((r['win'] for r in event_history)) + 2.5) / (len(event_history) + 5)
        out['event_count'] = math.log1p(len(event_history))
        out['prior_matches'] = math.log1p(len(completed))
        out['days_since_known_event_start'] = float((asof - completed[-1]['date']).days) if completed else float('nan')
        recent = [r for r in h if (asof - r['date']).days <= 90]
        out['retirement_proxy_90d'] = float(sum((r['retirement_proxy'] for r in recent)))
        out['known_matches_90d'] = float(len(recent))
        out['known_minutes_90d'] = sum((r['minutes'] for r in recent if math.isfinite(r['minutes'])))
        out.update(self.momentum.profile(player, surface, asof))
        out.update(self.charts.profile(player, surface, asof))
        return out

    def matchup(self, a, b, surface, asof, event='', level='A', best_of=3, round_name='R32', supplied_a=None, supplied_b=None):
        if a == b:
            raise ValueError('Choose two different players.')
        asof, surface = (tennis_date(asof), str(surface).title())
        if self.current_date != asof:
            self.advance(asof)
        features = self.sipko.pair_features(a, b, surface, asof)
        pa = self._player_context(a, surface, event, asof, supplied_a)
        pb = self._player_context(b, surface, event, asof, supplied_b)
        features.update({'diff_' + key: pa[key] - pb[key] for key in pa})
        for field in ('prior_matches', 'event_count'):
            features['coverage_min_' + field] = min(pa[field], pb[field])
        head = [r for r in self.history[a] if r['completed'] and r['opponent'] == b and (r['known_at'] < asof)]
        head = head[-20:]
        features['diff_h2h'] = 2 * ((sum((r['win'] for r in head)) + 1) / (len(head) + 2)) - 1
        features['coverage_h2h'] = math.log1p(len(head))
        for value in ('Hard', 'Clay', 'Grass', 'Carpet'):
            features['context_surface_' + value] = float(surface == value)
        features['context_surface_other'] = float(surface not in ('Hard', 'Clay', 'Grass', 'Carpet'))
        for value in ('A', 'M', 'G', 'F', 'D', 'O'):
            features['context_level_' + value] = float(level == value)
        features['context_level_other'] = float(level not in ('A', 'M', 'G', 'F', 'D', 'O'))
        for value in ('F', 'SF', 'QF', 'R16', 'R32', 'R64', 'R128', 'RR', 'BR'):
            features['context_round_' + value] = float(round_name == value)
        features['context_best_of'] = float(best_of)
        return features

def build_tennis_features(frame, config, charts, min_matches=10, progress=True):
    engine = TennisFeatureEngine(config, charts)
    rows, reverse, outcomes, meta = ([], [], [], [])
    began, last_message = (time.monotonic(), time.monotonic())
    done = 0
    for date, group in frame.groupby('date', sort=True):
        engine.advance(date)
        for row in group.to_dict('records'):
            w, l = (row['winner_id'], row['loser_id'])
            profiles = engine.profiles(w, l, row['surface'], date)
            count_w = sum((r['completed'] for r in engine.history[w]))
            count_l = sum((r['completed'] for r in engine.history[l]))
            if row['completed'] and date.year >= config.get('training_start', 2010) and (min(count_w, count_l) >= min_matches):
                a, b = sorted((w, l))
                side_a, side_b = ('winner', 'loser') if a == w else ('loser', 'winner')
                supplied_a = {f: row.get(side_a + '_' + f) for f in ('rank', 'rank_points', 'age', 'ht')}
                supplied_b = {f: row.get(side_b + '_' + f) for f in ('rank', 'rank_points', 'age', 'ht')}
                kwargs = dict(surface=row['surface'], asof=date, event=row.get('tourney_name', ''), level=str(row.get('tourney_level', 'A')), best_of=5 if row.get('best_of') == 5 else 3, round_name=str(row.get('round', 'R32')))
                x = engine.matchup(a, b, supplied_a=supplied_a, supplied_b=supplied_b, **kwargs)
                xr = engine.matchup(b, a, supplied_a=supplied_b, supplied_b=supplied_a, **kwargs)
                rows.append(x)
                reverse.append(xr)
                outcomes.append(int(a == w))
                meta.append({'match_id': row['match_id'], 'date': date, 'known_at': row['known_at'], 'tournament_id': str(row['tourney_id']), 'surface': row['surface'], 'level': str(row.get('tourney_level', '')), 'player_a_id': a, 'player_b_id': b, 'player_a': row[side_a + '_name'], 'player_b': row[side_b + '_name']})
            engine.schedule(row, profiles)
            done += 1
        now = time.monotonic()
        if progress and (now - last_message > 12 or done == len(frame)):
            print('  History %s: %s/%s rows; %s eligible matchups; %.0fs elapsed.' % (date.date(), done, len(frame), len(rows), now - began), flush=True)
            last_message = now
    if not rows:
        raise ValueError('No eligible training matches. Check years and player-history coverage.')
    x = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    xr = pd.DataFrame(reverse).reindex(columns=x.columns).replace([np.inf, -np.inf], np.nan)
    engine.advance(frame.known_at.max() + pd.Timedelta(days=1))
    return {'X': x, 'Xswap': xr, 'y': np.asarray(outcomes, dtype=int), 'metadata': pd.DataFrame(meta), 'engine_state': engine.export_state()}

def tennis_metrics(y, p):
    y, p = (np.asarray(y), np.asarray(p, dtype=float))
    if not len(y) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Cannot score empty data or invalid probabilities.')
    return {'matches': int(len(y)), 'correct': int(((p >= 0.5) == y).sum()), 'accuracy': float(accuracy_score(y, p >= 0.5)), 'log_loss': float(log_loss(y, p, labels=[0, 1])), 'brier': float(brier_score_loss(y, p)), 'auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None}

def tennis_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False, default=str))

def tennis_chart_store(args, frame):
    aliases, _ = tennis_aliases(frame)
    return ChartedShotStore(base_dir=args.charting_dir, research_enabled=args.research_charting, availability_path=args.chart_availability, player_aliases=aliases)

def train_tennis_model(args):
    out = Path(args.out)
    if (out / 'model.joblib').exists():
        raise ValueError('A model already exists here. Choose a new --out folder to preserve the completed run.')
    out.mkdir(parents=True, exist_ok=True)
    print('Loading ATP matches and constructing a shared training/prediction pipeline...', flush=True)
    frame, manifest = load_tennis_matches(args.data_dir, args.updates_dir)
    charts = tennis_chart_store(args, frame)
    for note in charts.notes:
        print('Charting: ' + note, flush=True)
    config = {'training_start': args.training_start, 'history_discount': args.history_discount, 'history_days': args.history_days, 'prior_points': args.prior_points}
    chart_manifest = {}
    if args.research_charting or args.chart_availability:
        chart_manifest = {str(p): tennis_sha(p) for p in sorted(Path(args.charting_dir).glob('charting-m-*.csv'))}
        if args.chart_availability:
            chart_manifest[str(args.chart_availability)] = tennis_sha(args.chart_availability)
    signature = {'inputs': manifest, 'charts': chart_manifest, 'config': config, 'chart_provenance': charts.provenance, 'source': tennis_sha(__file__)}
    cache = out / 'features.joblib'
    saved = joblib.load(cache) if cache.exists() else None
    if saved is not None and saved.get('signature') == signature:
        data = saved['data']
        print("Reusing this run's verified feature cache.", flush=True)
    else:
        data = build_tennis_features(frame, config, charts)
        joblib.dump({'signature': signature, 'data': data}, cache, compress=3)
    meta = data['metadata']
    train_end = pd.Timestamp('%s-01-01' % args.validation_year)
    print('Training integrated model; selecting the fixed mixture using %s only.' % args.selection_year, flush=True)
    estimator = estimator_fit(data['X'], data['Xswap'], data['y'], meta.date, meta.known_at, train_end=train_end, selection_year=args.selection_year, start=args.training_start, half_life=args.sample_half_life, iterations=args.iterations, threads=args.threads)
    validation = ((meta.date >= train_end) & (meta.known_at < pd.Timestamp('%s-01-01' % (args.validation_year + 1)))).to_numpy()
    if not validation.any():
        raise ValueError('No validation matches exist in the chosen year.')
    p = estimator_predict(estimator, data['X'].loc[validation], data['Xswap'].loc[validation])
    elo = 1 / (1 + 10 ** (-data['X'].loc[validation, 'diff_elo'].to_numpy() / 400))
    report = {'version': RESEARCH_VERSION, 'validation_year': args.validation_year, 'model': tennis_metrics(data['y'][validation], p), 'elo': tennis_metrics(data['y'][validation], elo), 'selection_year': args.selection_year, 'features': int(data['X'].shape[1]), 'history_configuration': config, 'limitations': RESEARCH_LIMITATIONS, 'chart_provenance': charts.provenance}
    prediction_rows = meta.loc[validation].copy()
    prediction_rows['outcome_a'] = data['y'][validation]
    prediction_rows['probability_a'] = p
    prediction_rows['elo_probability_a'] = elo
    prediction_rows.to_csv(out / 'validation_predictions.csv', index=False)
    tennis_json(out / 'validation_report.json', report)
    tennis_json(out / 'feature_schema.json', list(data['X'].columns))
    aliases, names = tennis_aliases(frame)
    bundle = {'version': RESEARCH_VERSION, 'estimator': estimator, 'config': config, 'history': frame, 'aliases': aliases, 'names': names, 'engine_state': data['engine_state'], 'chart_state': dict(vars(charts)), 'chart_configuration': {'base_dir': str(args.charting_dir), 'research_enabled': bool(args.research_charting), 'availability_path': str(args.chart_availability) if args.chart_availability else None}, 'chart_provenance': charts.provenance, 'chart_manifest': chart_manifest, 'trained_before': str(train_end.date()), 'mode': 'evaluated', 'evaluation': report, 'input_manifest': manifest, 'source_sha256': tennis_sha(__file__), 'limitations': RESEARCH_LIMITATIONS}
    joblib.dump(bundle, out / 'model.joblib', compress=3)
    tennis_json(out / 'training_details.json', {k: v for k, v in estimator.items() if k not in ('models', 'logistic', 'catboost', 'preprocessor') and isinstance(v, (str, int, float, bool, list, dict, type(None)))})
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    print('Saved predictor: %s' % (out / 'model.joblib'), flush=True)

class TennisPredictor:
    """Load once in a website backend and call predict for named matchups."""

    def __init__(self, bundle):
        if bundle.get('version') != RESEARCH_VERSION:
            raise ValueError('Unsupported model format; use its matching script.')
        if bundle.get('source_sha256') != tennis_sha(__file__):
            raise ValueError('Use the exact script version that trained this model; its feature definitions are frozen.')
        self.bundle = bundle
        self.engine = None
        self.position = 0
        self.records = bundle['history'].to_dict('records')

    @classmethod
    def load(cls, path):
        return cls(joblib.load(path))

    def resolve(self, name_or_id):
        value = str(name_or_id)
        if value in self.bundle['names']:
            return value
        resolved = self.bundle['aliases'].get(tennis_name(value))
        if resolved:
            return resolved
        import difflib
        suggestions = difflib.get_close_matches(tennis_name(value), list(self.bundle['aliases']), n=3, cutoff=0.45)
        raise ValueError('Unknown or ambiguous player %r. Possible names: %s' % (value, ', '.join(suggestions)))

    def _state(self, asof):
        if self.engine is None or (self.engine.current_date is not None and asof < self.engine.current_date):
            charts = ChartedShotStore()
            vars(charts).update(copy.deepcopy(self.bundle['chart_state']))
            state = self.bundle['engine_state']
            if asof >= state['current_date']:
                self.engine = TennisFeatureEngine.from_state(self.bundle['config'], charts, state)
                self.position = len(self.records)
            else:
                self.engine = TennisFeatureEngine(self.bundle['config'], charts)
                self.position = 0
        while self.position < len(self.records) and self.records[self.position]['date'] < asof:
            row = self.records[self.position]
            self.engine.advance(row['date'])
            self.engine.schedule(row)
            self.position += 1
        self.engine.advance(asof)
        return self.engine

    def predict(self, player_a, player_b, surface, as_of, event='', level='A', best_of=3, round_name='R32', rank_a=None, rank_b=None, rank_points_a=None, rank_points_b=None):
        asof = tennis_date(as_of)
        if asof < tennis_date(self.bundle['trained_before']):
            raise ValueError("Prediction date precedes this model's training cutoff. Use a model fitted before that date.")
        surface = str(surface).title()
        if surface not in ('Hard', 'Clay', 'Grass', 'Carpet') or best_of not in (3, 5):
            raise ValueError('Use Hard, Clay, Grass or Carpet, and best_of 3 or 5.')
        a, b = (self.resolve(player_a), self.resolve(player_b))
        engine = self._state(asof)
        counts = [sum((r['completed'] for r in engine.history[player])) for player in (a, b)]
        if min(counts) < 10:
            raise ValueError('Both players need at least 10 completed, available historical matches; found %s and %s.' % tuple(counts))
        common = dict(surface=surface, asof=asof, event=event, level=level, best_of=best_of, round_name=round_name)
        ma, mb = ({'rank': rank_a, 'rank_points': rank_points_a}, {'rank': rank_b, 'rank_points': rank_points_b})
        x = engine.matchup(a, b, supplied_a=ma, supplied_b=mb, **common)
        xr = engine.matchup(b, a, supplied_a=mb, supplied_b=ma, **common)
        p = float(estimator_predict(self.bundle['estimator'], pd.DataFrame([x]), pd.DataFrame([xr]))[0])
        names = self.bundle['names']
        pa, pb = engine.profiles(a, b, surface, asof)
        most_recent = engine.latest_known_tournament
        return {'player_a': names[a], 'player_b': names[b], 'probability_a': p, 'probability_b': 1 - p, 'predicted_winner': names[a] if p >= 0.5 else names[b], 'forecast_date': str(asof.date()), 'surface': surface, 'event': event, 'event_level': level, 'best_of': best_of, 'round': round_name, 'model_mode': self.bundle['mode'], 'trained_before': self.bundle['trained_before'], 'latest_available_tournament_start': str(most_recent.date()) if most_recent is not None else None, 'days_since_latest_available_tournament_start': int((asof - most_recent).days) if most_recent is not None else None, 'historical_rates': {'player_a': {k: pa[k] for k in ('serve_won', 'return_won')}, 'player_b': {k: pb[k] for k in ('serve_won', 'return_won')}}, 'coverage': {k: float(v) for k, v in x.items() if k.startswith('coverage_') and math.isfinite(float(v))}, 'chart_provenance': self.bundle['chart_provenance'], 'limitations': self.bundle['limitations']}

def finalize_tennis_model(args):
    source = Path(args.model)
    out = Path(args.output)
    if source.resolve() == out.resolve() or out.exists():
        raise ValueError('Choose a new output filename to preserve the evaluated model.')
    bundle = joblib.load(source)
    cache_path = source.parent / 'features.joblib'
    if not cache_path.exists():
        raise ValueError('The original features.joblib must remain beside the evaluated model.')
    saved = joblib.load(cache_path)
    signature = saved['signature']
    expected = {'source': bundle['source_sha256'], 'inputs': bundle['input_manifest'], 'config': bundle['config'], 'charts': bundle['chart_manifest'], 'chart_provenance': bundle['chart_provenance']}
    if any((signature.get(key) != value for key, value in expected.items())):
        raise ValueError('Feature cache does not match this model.')
    if tennis_sha(__file__) != bundle['source_sha256']:
        raise ValueError('Use the frozen script that trained this model for deployment refitting.')
    data, cutoff = (saved['data'], tennis_date(args.as_of))
    if cutoff <= tennis_date(bundle['trained_before']):
        raise ValueError('Deployment refit date must follow the original training cutoff.')
    meta = data['metadata']
    print('Refitting the saved mixture using results available before %s.' % cutoff.date(), flush=True)
    bundle['estimator'] = refit_selected(bundle['estimator'], data['X'], data['Xswap'], data['y'], meta.date, meta.known_at, train_end=cutoff, start=bundle['config']['training_start'], half_life=args.sample_half_life)
    bundle['mode'] = 'deployment_refit'
    bundle['trained_before'] = str(cutoff.date())
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out, compress=3)
    print('Saved %s. Historical evaluation scores remain attached to the earlier evaluated model.' % out, flush=True)

def tennis_main(argv=None):
    parser = argparse.ArgumentParser(description='Paper-informed ATP model: integrated training and named-player prediction.')
    actions = parser.add_subparsers(dest='action', required=True)
    train = actions.add_parser('train', help='Train one integrated predictor; select its mixture in 2024 and report 2025.')
    train.add_argument('--data-dir', default='data/raw')
    train.add_argument('--updates-dir', default='data/external/atp_updates')
    train.add_argument('--charting-dir', default='data/external/match_charting')
    train.add_argument('--chart-availability', help='CSV containing match_id and actual known publication timestamp.')
    train.add_argument('--research-charting', action='store_true', help='Allow charts without verified publication timing; exploratory only.')
    train.add_argument('--out', default='models/research_v5')
    train.add_argument('--training-start', type=int, default=2010)
    train.add_argument('--sample-half-life', type=float, default=5.0)
    train.add_argument('--history-discount', type=float, default=0.8)
    train.add_argument('--history-days', type=int, default=1826)
    train.add_argument('--prior-points', type=float, default=50.0)
    train.add_argument('--selection-year', type=int, default=2024)
    train.add_argument('--validation-year', type=int, default=2025)
    train.add_argument('--iterations', type=int, default=500)
    train.add_argument('--threads', type=int, default=4)
    predict = actions.add_parser('predict', help='Predict a named matchup using the same causal feature pipeline.')
    predict.add_argument('--model', default='models/research_v5/model.joblib')
    predict.add_argument('--player-a', required=True)
    predict.add_argument('--player-b', required=True)
    predict.add_argument('--surface', required=True, choices=['Hard', 'Clay', 'Grass', 'Carpet'])
    predict.add_argument('--date', required=True, help='Forecast date YYYY-MM-DD, not a fabricated exact match start.')
    predict.add_argument('--event', default='')
    predict.add_argument('--level', default='A', choices=['A', 'M', 'G', 'F', 'D', 'O'])
    predict.add_argument('--best-of', type=int, default=3, choices=[3, 5])
    predict.add_argument('--round', default='R32', dest='round_name')
    predict.add_argument('--rank-a', type=float)
    predict.add_argument('--rank-b', type=float)
    predict.add_argument('--rank-points-a', type=float)
    predict.add_argument('--rank-points-b', type=float)
    finalize = actions.add_parser('finalize', help='Refit the selected mixture on all known results for use in a website.')
    finalize.add_argument('--model', default='models/research_v5/model.joblib')
    finalize.add_argument('--output', default='models/research_v5/deployment.joblib')
    finalize.add_argument('--as-of', required=True)
    finalize.add_argument('--sample-half-life', type=float, default=5.0)
    args = parser.parse_args(argv)
    try:
        if args.action == 'train':
            if not 1991 <= args.training_start < args.selection_year < args.validation_year <= 2100:
                raise ValueError('Require training start < selection year < validation year.')
            if args.validation_year != args.selection_year + 1:
                raise ValueError('Validation year must immediately follow the inner selection year.')
            if not all((math.isfinite(v) for v in (args.history_discount, args.prior_points, args.sample_half_life))) or not 0 < args.history_discount <= 1 or args.history_days < 1 or (args.prior_points <= 0) or (args.sample_half_life <= 0):
                raise ValueError('Use positive history/weight settings and a history discount no greater than 1.')
            if args.iterations < 1 or args.threads < 1:
                raise ValueError('Iterations and threads must be positive integers.')
            train_tennis_model(args)
        elif args.action == 'predict':
            model = TennisPredictor.load(args.model)
            result = model.predict(args.player_a, args.player_b, args.surface, args.date, event=args.event, level=args.level, best_of=args.best_of, round_name=args.round_name, rank_a=args.rank_a, rank_b=args.rank_b, rank_points_a=args.rank_points_a, rank_points_b=args.rank_points_b)
            print(json.dumps(result, indent=2, allow_nan=False))
        else:
            finalize_tennis_model(args)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        parser.exit(2, 'Tennis model: %s\n' % exc)
if __name__ == '__main__':
    tennis_main()
