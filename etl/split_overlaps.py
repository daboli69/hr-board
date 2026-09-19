"""Daily, venue-specific post-ASG research. No other board signals enter scoring."""
from datetime import datetime, timedelta
import math
import requests

VERSION = 'venue-overlap-1'
DEFAULT_THRESHOLDS = {'pa': 50, 'bf': 30}
METRICS = {
    'HR_OVERLAP': {'batter': ['iso', 'slg', 'fb_pct', 'barrel_pct', 'ev', 'hr_rate'],
                   'pitcher': ['hr9', 'hr_fb', 'iso', 'slg', 'fb_pct', 'barrel_pct', 'ev']},
    'HIT_OVERLAP': {'batter': ['avg', 'obp', 'woba', 'k_pct', 'babip', 'contact_pct'],
                    'pitcher': ['avg', 'obp', 'whip', 'babip', 'k_pct', 'contact_pct']},
    'HRR_OVERLAP': {'batter': ['h', 'r', 'rbi', 'avg', 'obp', 'slg', 'ops', 'woba', 'hr'],
                    'pitcher': ['h', 'obp', 'slg', 'ops', 'hr', 'whip', 'baserunners']},
}
# Two complementary rate statistics per side. Fixed reference scales, never fit to outcomes.
SCORING = {'HR_OVERLAP': [('iso', .170), ('barrel_pct', 8)],
           'HIT_OVERLAP': [('avg', .250), ('contact_pct', 75)],
           'HRR_OVERLAP': [('obp', .320), ('iso', .170)]}


def ratio(n, d, scale=1):
    return round(scale * n / d, 5) if d else None


def get_json(path, params=None):
    response = requests.get('https://statsapi.mlb.com/api/v1/' + path, params=params, timeout=25)
    response.raise_for_status()
    return response.json()


def cutoff_for(date):
    schedule = get_json('schedule', {'sportId': 1, 'season': int(date[:4]), 'gameType': 'A'})
    dates = [g.get('officialDate', d['date']) for d in schedule.get('dates', [])
             for g in d.get('games', []) if g.get('gameType') == 'A']
    if not dates:
        raise ValueError('Official All-Star date unavailable')
    return (datetime.fromisoformat(max(dates)) + timedelta(days=1)).date().isoformat()


def game_logs(ids, group, season):
    result = {}
    ids = sorted(set(int(i) for i in ids if i))
    for offset in range(0, len(ids), 40):
        data = get_json('people', {'personIds': ','.join(map(str, ids[offset:offset+40])),
                                  'hydrate': f'stats(group=[{group}],type=[gameLog],season={season})'})
        for person in data.get('people', []):
            result[person['id']] = [row for block in person.get('stats', [])
                                    if block.get('group', {}).get('displayName') == group
                                    for row in block.get('splits', [])]
    return result


def split_stats(logs, pitches, home, start, date, pitching=False):
    """Select one venue before aggregation; exclude today to prevent live leakage."""
    rows = [r for r in logs if r.get('gameType') == 'R' and r.get('isHome') is home
            and start <= r.get('date', '') < date]
    stats = [r['stat'] for r in rows]
    total = lambda key: sum(float(s.get(key, 0) or 0) for s in stats)
    h, ab, bb, hbp = (total(k) for k in ('hits', 'atBats', 'baseOnBalls', 'hitByPitch'))
    hr, doubles, triples, k, sf = (total(x) for x in ('homeRuns', 'doubles', 'triples', 'strikeOuts', 'sacFlies'))
    pa = total('battersFaced' if pitching else 'plateAppearances')
    outs = sum(int(str(s.get('inningsPitched', '0')).split('.')[0]) * 3 +
               int((str(s.get('inningsPitched', '0')).split('.') + ['0'])[1]) for s in stats) if pitching else 0
    tb = h + doubles + 2 * triples + 3 * hr
    result = {'pa': None if pitching else pa, 'bf': pa if pitching else None,
              'ip': round(outs / 3, 4) if pitching else None, 'games': len(rows),
              'h': h, 'r': total('runs'), 'rbi': total('rbi'), 'hr': hr,
              'avg': ratio(h, ab), 'obp': ratio(h+bb+hbp, ab+bb+hbp+sf),
              'slg': ratio(tb, ab), 'iso': ratio(tb-h, ab), 'k_pct': ratio(k, pa, 100),
              'babip': ratio(h-hr, ab-k-hr+sf), 'hr_rate': ratio(hr, pa, 100),
              'whip': ratio(h+bb, outs, 3), 'hr9': ratio(hr, outs, 27),
              'era': ratio(total('earnedRuns'), outs, 27), 'baserunners': h+bb+hbp}
    result['ops'] = round(result['obp'] + result['slg'], 5) if result['obp'] is not None and result['slg'] is not None else None
    # Raw Statcast is already pulled once by the board; compute only this venue.
    half = ('Top' if home else 'Bot') if pitching else ('Bot' if home else 'Top')
    p = pitches[(pitches.inning_topbot == half) & (pitches.game_date.astype(str).str[:10] >= start)
                & (pitches.game_date.astype(str).str[:10] < date)]
    bbe = p[p.type == 'X']
    fb = int((bbe.bb_type == 'fly_ball').sum())
    result.update({'bbe': len(bbe), 'fb_pct': ratio(fb, len(bbe), 100),
                   'barrel_pct': ratio(int((bbe.launch_speed_angle == 6).sum()), int(bbe.launch_speed_angle.notna().sum()), 100),
                   'ev': float(bbe.launch_speed.mean()) if bbe.launch_speed.notna().any() else None,
                   'hr_fb': ratio(int((p.events == 'home_run').sum()), fb, 100)})
    swings = p.description.isin(['swinging_strike', 'swinging_strike_blocked', 'missed_bunt',
                                'foul', 'foul_tip', 'foul_bunt', 'hit_into_play', 'bunt_foul_tip'])
    miss = p.description.isin(['swinging_strike', 'swinging_strike_blocked', 'missed_bunt'])
    result['contact_pct'] = ratio(int(swings.sum()-miss.sum()), int(swings.sum()), 100)
    ended = p[p.events.notna()]
    w = ended[ended.woba_value.notna() & ended.woba_denom.notna()]
    result['woba'] = ratio(float(w.woba_value.sum()), float(w.woba_denom.sum()))
    result['statcast_pa'] = len(ended)
    return result


def recent_mix(pitches, logs, date):
    starts = sorted({(r['date'], r['game']['gamePk']) for r in logs
                     if r.get('gameType') == 'R' and r.get('date', '') < date
                     and r.get('stat', {}).get('gamesStarted') == 1}, reverse=True)[:5]
    p = pitches[pitches.game_pk.isin([pk for _, pk in starts]) & (pitches.game_date.astype(str).str[:10] < date)]
    p = p[p.pitch_type.notna()]
    rows = []
    for pitch, group in p.groupby('pitch_type'):
        rows.append({'pitch_type': str(pitch), 'usage_pct': ratio(len(group), len(p), 100),
                     **{f'usage_pct_vs_{hand}': ratio(int((group.stand == hand).sum()), int((p.stand == hand).sum()), 100)
                        for hand in ('R', 'L')}})
    return {'starts': [{'date': d, 'game_pk': pk} for d, pk in starts],
            'starts_count': len(starts), 'covered_starts': int(p.game_pk.nunique()),
            'pitches': len(p), 'rows': sorted(rows, key=lambda r: -r['usage_pct'])}


def strength(stats, components):
    values = [stats.get(k) for k, _ in components]
    if any(v is None or not math.isfinite(v) for v in values):
        return None
    return sum(min(2, max(0, v / scale)) for v, (_, scale) in zip(values, components)) / len(values)


def rank_rows(rows, thresholds):
    qualified = [r for r in rows if r['overlap_strength'] is not None and
                 r['batter_stats']['pa'] >= thresholds['pa'] and r['pitcher_stats']['bf'] >= thresholds['bf']]
    return [dict(r, rank=i+1, thresholds=dict(thresholds)) for i, r in enumerate(
        sorted(qualified, key=lambda r: (-r['overlap_strength'], r['game_pk'], r['id'])))]


def build(board, frame, cutoff=None, batter_logs=None, pitcher_logs=None):
    date = board['slate_date']
    start = cutoff or cutoff_for(date)
    payload = {'schema_version': 1, 'model_version': VERSION, 'slate_date': date,
               'generated_at': board['generated_at'], 'window_start': start, 'window_end_exclusive': date,
               'status': 'FRESH', 'default_thresholds': dict(DEFAULT_THRESHOLDS), 'metrics': METRICS,
               'formula': '100 × batter component × pitcher component; each component is the mean of two rate/reference ratios capped at 2. Research strength, not probability.',
               'scoring': SCORING, 'boards': {key: [] for key in METRICS}}
    if date <= start:
        payload.update(status='NO_POST_ASG_SAMPLE', note='Post-All-Star split samples begin after the break.')
        return payload
    players = board.get('players', [])
    batter_logs = batter_logs if batter_logs is not None else game_logs([p['id'] for p in players], 'hitting', int(date[:4]))
    pitcher_logs = pitcher_logs if pitcher_logs is not None else game_logs([(p.get('opp_pitcher') or {}).get('id') for p in players], 'pitching', int(date[:4]))
    cache, mixes = {}, {}
    for p in players:
        pid = (p.get('opp_pitcher') or {}).get('id')
        if not pid or p.get('side') not in ('home', 'away'):
            continue
        home = p['side'] == 'home'
        for ident, is_home, pitching, logs in [(p['id'], home, False, batter_logs), (pid, not home, True, pitcher_logs)]:
            key = (ident, is_home, pitching)
            if key not in cache:
                column = 'pitcher' if pitching else 'batter'
                cache[key] = split_stats(logs.get(ident, []), frame[frame[column] == ident], is_home, start, date, pitching)
        if pid not in mixes:
            mixes[pid] = recent_mix(frame[frame.pitcher == pid], pitcher_logs.get(pid, []), date)
        b, a = cache[(p['id'], home, False)], cache[(pid, not home, True)]
        confidence = round(100 * min(b['pa'] / (b['pa'] + 100), a['bf'] / (a['bf'] + 100)), 1)
        for family, components in SCORING.items():
            bs, ps = strength(b, components), strength(a, components)
            row = {'id': p['id'], 'name': p['name'], 'game_pk': p['game_pk'],
                   'game': f"{p.get('team')} vs {p.get('opp_team')}", 'lineup_spot': p.get('lineup_spot'),
                   'lineup_status': p.get('lineup_status'), 'pitcher_id': pid, 'pitcher_name': p['opp_pitcher'].get('name'),
                   'split_label': f"{p.get('team')} Batter {'HOME' if home else 'AWAY'} vs {p.get('opp_team')} Pitcher {'AWAY' if home else 'HOME'}",
                   'signal_family': 'SPLIT_' + family, 'model_version': VERSION,
                   'overlap_strength': round(100 * bs * ps, 2) if bs is not None and ps is not None else None,
                   'batter_strength': bs, 'pitcher_vulnerability': ps, 'sample_confidence': confidence,
                   'batter_stats': b, 'pitcher_stats': a,
                   'key_batter_evidence': {k: b[k] for k, _ in components},
                   'key_pitcher_evidence': {k: a[k] for k, _ in components},
                   'pitcher_recent_pitch_mix': mixes[pid], 'actual_outcome': None}
            payload['boards'][family].append(row)
    payload['rankings'] = {k: [{'id': r['id'], 'game_pk': r['game_pk'], 'rank': r['rank'], 'strength': r['overlap_strength']} for r in rank_rows(v, DEFAULT_THRESHOLDS)] for k, v in payload['boards'].items()}
    return payload


if __name__ == '__main__':
    # Targeted verification/refresh of this feature; normal cron calls build() above.
    import argparse
    import json
    from pathlib import Path
    from etl import statcast_data
    parser = argparse.ArgumentParser()
    parser.add_argument('--board', default='docs/board.json')
    parser.add_argument('--output', help='Write the feature payload here without altering the existing board')
    parser.add_argument('--publish', help='Attach a validated payload to the same dated board without recomputing other features')
    args = parser.parse_args()
    board = json.loads(Path(args.board).read_text(encoding='utf-8'))
    if args.publish:
        result = json.loads(Path(args.publish).read_text(encoding='utf-8'))
        if result['slate_date'] != board['slate_date'] or result['generated_at'] != board['generated_at']:
            parser.error('Payload must match this exact board generation')
        if result.get('status') != 'FRESH' or set(result.get('boards', {})) != set(METRICS):
            parser.error('Only a complete, successful feature payload can be published')
        board['split_overlaps'] = result
        temporary = Path(args.board).with_suffix('.tmp')
        temporary.write_text(json.dumps(board, allow_nan=False), encoding='utf-8')
        temporary.replace(args.board)
        from etl.pregame_records import freeze_games
        snapshot = {'date': board['slate_date'], 'split_overlaps': result,
                    'players': [{k: p.get(k) for k in ('id', 'name', 'game_pk')} for p in board['players']]}
        frozen = freeze_games(snapshot, board['games'], Path(args.board).parent / 'snapshots',
                              board['generated_at'], split_only=True)
        print(f'Published split payload; other board fields unchanged; froze {frozen} split game records')
        raise SystemExit(0)
    if not args.output:
        parser.error('--output or --publish is required')
    frame = statcast_data.pull_season(board['slate_date'][:4] + '-03-01', board['slate_date'])
    result = build(board, frame)
    Path(args.output).write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'date': result['slate_date'],
                      'counts': {k: len(v) for k, v in result['boards'].items()}}))
