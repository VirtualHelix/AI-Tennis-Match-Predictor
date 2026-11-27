"""Pure descriptive-history merge; never updates the saved prediction engine.

Current-season replacement requires a complete, contiguous provider backfill.
Until then only clearly newer results enter recent cards, and H2H totals remain
the saved-history totals. Names are matched conservatively; provider identifiers
are never treated as local player identifiers without an explicit valid alias.
"""
from collections import defaultdict
import copy
from datetime import date, datetime, timedelta, timezone
import math
import re
import unicodedata

from courtiq.sportstats import parse_completed_score


_STATUS_FIELDS = (
    'configured', 'state', 'last_sync', 'coverage_complete', 'covered_ranges',
    'coverage_start', 'coverage_end', 'coverage_year', 'coverage_scope', 'matches',
    'requests_today', 'request_budget', 'quota_remaining', 'quota_reset', 'error',
    'cache_warning', 'delay_seconds', 'data_delay_seconds', 'source_url',
    'progress', 'refresh_interval_seconds',
)


def _text(value):
    if not isinstance(value, str):
        return None
    result = value.strip()
    if not result or len(result) > 500 or any(ord(c) < 32 for c in result):
        return None
    return result if result.lower() not in ('nan', 'none', 'nat', '<na>') else None


def _day(value):
    try:
        if isinstance(value, (date, datetime)):
            return date(int(value.year), int(value.month), int(value.day))
        if isinstance(value, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
            return date.fromisoformat(value)
    except (ValueError, TypeError, OverflowError):
        pass
    return None


def _stamp(value):
    try:
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None:
            return value.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError):
        pass
    return None


def _opaque(value):
    return value if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.:-]{1,100}', value) else None


def _tokens(value):
    value = _text(value)
    if value is None:
        return ()
    value = unicodedata.normalize('NFKD', value.casefold())
    value = ''.join(c for c in value if not unicodedata.combining(c) and c not in "'’‘`")
    return tuple(''.join(c if c.isalnum() else ' ' for c in value).split())


class _PlayerResolver:
    def __init__(self, names, aliases):
        self.names = {str(key): value for key, value in names.items() if _text(value)}
        self.exact, self.initials = defaultdict(set), defaultdict(set)
        for player, display in self.names.items():
            full = _tokens(display)
            if not full:
                continue
            self.exact[full].add(player)
            for count in range(1, len(full)):
                signature = tuple(token[0] for token in full[:count]) + full[count:]
                self.initials[signature].add(player)
        self.aliases = aliases if isinstance(aliases, dict) else {}

    def participant(self, raw):
        if not isinstance(raw, dict):
            return None
        provider, display = _opaque(raw.get('provider_id')), _text(raw.get('name'))
        if provider is None or display is None:
            return None
        tokens = _tokens(display)
        if not tokens:
            return None
        matched = None
        if provider in self.aliases:
            # An invalid explicit alias fails closed, rather than selecting some
            # other player by a weaker name rule or creating a new local ID.
            alias = self.aliases[provider]
            if isinstance(alias, str) and alias in self.names:
                matched = alias
        else:
            initial_count = 0
            for token in tokens:
                if len(token) != 1:
                    break
                initial_count += 1
            candidates = set(self.exact.get(tokens, ()))
            if 0 < initial_count < len(tokens):
                candidates.update(self.initials.get(tokens, ()))
            if len(candidates) == 1:
                matched = next(iter(candidates))
        return {'id':matched or 'sportstats-player:' + provider,
                'name':self.names[matched] if matched else display, 'provider_id':provider}


def _api_records(snapshot, resolver, now, limit):
    grouped = defaultdict(list)
    raw_records = snapshot.get('records', []) if isinstance(snapshot, dict) else []
    if not isinstance(raw_records, (list, tuple)):
        return []
    for raw in raw_records:
        if not isinstance(raw, dict) or raw.get('source') != 'sportstats' or raw.get('date_basis') != 'match_date':
            continue
        mid = raw.get('match_id')
        if not isinstance(mid, str) or not mid.startswith('sportstats:') or not _opaque(mid[10:]):
            continue
        day, observed = _day(raw.get('match_date')), _stamp(raw.get('observed_at'))
        if (day is None or observed is None or observed > now or observed.date() < day
                or day > limit or day.year != now.year):
            continue
        winner, loser = resolver.participant(raw.get('winner')), resolver.participant(raw.get('loser'))
        score = _text(raw.get('score'))
        if (winner is None or loser is None or winner['id'] == loser['id']
                or winner['provider_id'] == loser['provider_id'] or score is None
                or raw.get('score_kind') not in ('games', 'sets')
                or re.search(r'W/O|WALKOVER|RET|DEF|ABD', score, flags=re.I)):
            continue
        surface = _text(raw.get('surface'))
        surface = surface.title() if surface else None
        row = {'match_id':mid, 'match_date':day.isoformat(), 'tournament_date':None,
               'winner':winner, 'loser':loser, 'event':_text(raw.get('event')),
               'surface':surface if surface in ('Hard', 'Clay', 'Grass', 'Carpet') else None,
               'score':score, 'score_kind':raw['score_kind'], 'source':'sportstats',
               # Validate the full score again at this merge boundary. Cached
               # aggregate set counts are never individual game scores, and
               # unverified caller-provided arrays cannot override the score.
               'set_scores':parse_completed_score(score) if raw['score_kind'] == 'games' else [],
               'date_basis':'match_date', 'observed_at':observed.isoformat().replace('+00:00', 'Z'),
               '_day':day, '_tie':('', 0., mid), '_observed':observed}
        grouped[mid].append(row)
    result = []
    for candidates in grouped.values():
        latest = max(row['_observed'] for row in candidates)
        newest = [row for row in candidates if row['_observed'] == latest]
        # An equally recent contradictory duplicate is unknown, not a second
        # match or an arbitrary winner chosen by input order.
        if all(row == newest[0] for row in newest[1:]):
            result.append(newest[0])
    return result


def _legacy_records(records, names, asof, limit):
    result, seen = [], set()
    for raw in records:
        if not isinstance(raw, dict) or not raw.get('completed'):
            continue
        day, known = _day(raw.get('date')), _day(raw.get('known_at'))
        mid = raw.get('match_id')
        if day is None or known is None or not known < asof or day > limit or mid is None or str(mid) in seen:
            continue
        winner, loser = str(raw.get('winner_id', '')), str(raw.get('loser_id', ''))
        if not winner or not loser or winner == loser:
            continue
        try:
            number = float(raw.get('match_num', 0))
            number = number if math.isfinite(number) else 0.
        except (ValueError, TypeError):
            number = 0.
        result.append({'match_id':str(mid), 'tournament_date':day.isoformat(),
            'winner':{'id':winner, 'name':names.get(winner) or _text(raw.get('winner_name')) or winner},
            'loser':{'id':loser, 'name':names.get(loser) or _text(raw.get('loser_name')) or loser},
            'event':_text(raw.get('tourney_name')), 'surface':_text(raw.get('surface')),
            'score':_text(raw.get('score')), 'source':'saved_history', 'date_basis':'tournament_start',
            'set_scores':parse_completed_score(raw.get('score')),
            '_day':day, '_tie':(str(raw.get('tourney_id', '')), number, str(mid))})
        seen.add(str(mid))
    return result


def _complete_year(status, today):
    if status.get('coverage_complete') is not True:
        return False
    ranges = []
    provided = status.get('covered_ranges', [])
    if not isinstance(provided, (list, tuple)):
        return False
    for entry in provided:
        if not isinstance(entry, dict):
            continue
        start, end = _day(entry.get('start')), _day(entry.get('end'))
        if start is not None and end is not None and start <= end:
            ranges.append((start, min(end, today)))
    cursor = date(today.year, 1, 1)
    for start, end in sorted(ranges):
        if end < cursor:
            continue
        if start > cursor:
            return False
        cursor = end + timedelta(days=1)
        if cursor > today:
            return True
    return False


def _public(row):
    return copy.deepcopy({key:value for key, value in row.items() if not key.startswith('_')})


def _recent_view(template, rows, names, a, b):
    result = copy.deepcopy(template)
    pair = [row for row in rows if {row['winner']['id'], row['loser']['id']} == {a, b}]
    result['head_to_head'] = [_public(row) for row in pair[:3]]
    result['players'] = []
    for player in (a, b):
        selected = [row for row in rows if player in (row['winner']['id'], row['loser']['id'])][:3]
        matches = []
        for row in selected:
            item = _public(row)
            won = item['winner']['id'] == player
            item.update(result='W' if won else 'L', opponent=copy.deepcopy(item['loser' if won else 'winner']))
            matches.append(item)
        result['players'].append({'id':player, 'name':names.get(player, player), 'matches':matches})
    return result, pair


def merge_results_history(legacy_records, names, args, model_data_through,
                          old_recent, old_h2h, feed_snapshot, now=None, aliases=None):
    """Return independent ``(recent_matches, head_to_head)`` response objects.

    ``now`` is an optional aware datetime (or ISO timestamp) for deterministic
    tests. Historical forecast dates retain the saved view, since current feed
    observations cannot reconstruct what was knowable on those earlier dates.
    """
    now = datetime.now(timezone.utc) if now is None else _stamp(now)
    if now is None:
        raise ValueError('now must be a timestamp with an explicit timezone')
    asof = _day(args.get('as_of'))
    if asof is None:
        raise ValueError('as_of must be a valid YYYY-MM-DD date')
    recent, h2h = copy.deepcopy(old_recent), copy.deepcopy(old_h2h)
    if asof < now.date():
        return recent, h2h
    snapshot = feed_snapshot if isinstance(feed_snapshot, dict) else {}
    status = snapshot.get('status') if isinstance(snapshot.get('status'), dict) else {}
    public_status = {key:copy.deepcopy(status[key]) for key in _STATUS_FIELDS if key in status}
    resolver = _PlayerResolver(names, aliases)
    api = _api_records(snapshot, resolver, now, min(asof, now.date()))
    for target in (recent, h2h):
        target['feed'] = copy.deepcopy(public_status)
    if not api:
        note = 'No usable current-season feed records are available for this view; saved-history records and totals are unchanged.'
        for target in (recent, h2h):
            target['coverage_note'] = note
        return recent, h2h

    a, b = str(args['player_a']), str(args['player_b'])
    legacy = _legacy_records(legacy_records, resolver.names, asof, now.date())
    complete = _complete_year(status, now.date())
    date_note = ('API rows use match dates; saved rows use tournament-start dates. Scores are winner-first. '
                 'Same-date order uses record identifiers, not verified match times. These are recorded dataset totals, not official all-time totals.')
    if any(row['surface'] is None for row in api):
        date_note += ' Feed rows without a recorded surface are excluded from surface-specific totals.'
    if any(row[side]['id'].startswith('sportstats-player:') for row in api for side in ('winner', 'loser')):
        date_note += ' Unmatched feed identities cannot be assigned to saved players for H2H totals.'
    if complete:
        combined = [row for row in legacy if row['_day'].year < now.year] + api
        note = ('The %s completed ATP singles main-draw feed replaces all saved descriptive records from that season; '
                'competition coverage can differ. Earlier saved years are retained. ' % now.year) + date_note
    else:
        last_legacy = max((row['_day'] for row in legacy), default=None)
        threshold = last_legacy + timedelta(days=21) if last_legacy else None
        overlay = [row for row in api if threshold is None or row['_day'] > threshold]
        combined = legacy + overlay
        note = ('Current-season backfill is incomplete. H2H totals and its latest meeting remain the saved-history values. '
                'Recent cards may include newer feed results only after %s; the intervening period may be incomplete. '
                % (threshold.isoformat() if threshold else 'the available saved history')) + date_note
        if not overlay:
            for target in (recent, h2h):
                target['coverage_note'] = note
            return recent, h2h

    combined.sort(key=lambda row:(row['_day'], row['_tie']), reverse=True)
    recent, pair = _recent_view(recent, combined, resolver.names, a, b)
    latest_day = max((row['_day'] for row in combined), default=_day(model_data_through))
    through = latest_day.isoformat() if latest_day else model_data_through
    recent.update(source='saved_history+sportstats', date_basis='mixed', data_through=through,
                  model_data_through=model_data_through, coverage_note=note, scope=note)
    if complete:
        def totals(rows):
            wins = sum(row['winner']['id'] == a for row in rows)
            return {'played':len(rows), 'wins_a':wins, 'wins_b':len(rows) - wins}
        latest = pair[0] if pair else None
        h2h.update(year=asof.year, overall=totals(pair),
                   ytd=totals([row for row in pair if row['_day'].year == asof.year]),
                   surface=dict(totals([row for row in pair if row['surface'] == args['surface']]), surface=args['surface']),
                   source='saved_history+sportstats', data_through=through, date_basis='mixed', scope=note,
                   history_start=min((row['_day'] for row in combined), default=None))
        h2h['history_start'] = h2h['history_start'].isoformat() if h2h['history_start'] else None
        h2h['latest_meeting'] = ({'date':latest['_day'].isoformat(), 'event':latest['event'],
            'surface':latest['surface'], 'winner_id':latest['winner']['id'], 'winner_name':latest['winner']['name'],
            'score':latest['score'], 'source':latest['source'], 'date_basis':latest['date_basis'],
            'set_scores':copy.deepcopy(latest['set_scores']),
            'match_date':latest.get('match_date'), 'tournament_date':latest.get('tournament_date'),
            'score_kind':latest.get('score_kind')} if latest else None)
    h2h.update(source='saved_history+sportstats', model_data_through=model_data_through,
               totals_source='saved_history+sportstats' if complete else 'saved_history', coverage_note=note)
    return recent, h2h
