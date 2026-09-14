"""Grade immutable research against official final box scores, separately from legacy history."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import requests
from etl.pregame_records import _utc


def grade_record(record, schedule, box):
    game_id = record['game_pk']
    games = [g for day in schedule.get('dates', []) for g in day.get('games', []) if g.get('gamePk') == game_id]
    if not games or games[0].get('status', {}).get('abstractGameState') != 'Final':
        return None
    captured, generated, start = (_utc(record.get(k)) for k in ['captured_at', 'generated_at', 'kickoff'])
    if not captured or not generated or not start or not generated <= captured < start:
        raise ValueError('Record does not establish a pregame observation')
    confirmed_bench = set()
    for side in ('home', 'away'):
        team = box.get('teams', {}).get(side, {})
        appearances = team.get('teamStats', {}).get('batting', {}).get('plateAppearances')
        if not team.get('players') or not isinstance(appearances, (int, float)) or appearances <= 0:
            raise ValueError(f'Final {side} boxscore is incomplete; retry required')
        counted = sum(p.get('stats', {}).get('batting', {}).get('plateAppearances', 0)
                      for p in team['players'].values())
        if counted == appearances and isinstance(team.get('batters'), list):
            batters = {str(pid) for pid in team['batters']}
            for p in team['players'].values():
                pid = str(p.get('person', {}).get('id'))
                if pid not in batters and p.get('gameStatus', {}).get('isOnBench') is True and not p.get('stats', {}).get('batting'):
                    confirmed_bench.add(pid)
    actual = {str(p.get('person', {}).get('id')): p for team in box.get('teams', {}).values() for p in team.get('players', {}).values()}
    outcomes = []
    for player in record.get('players', []):
        batting = actual.get(str(player['id']), {}).get('stats', {}).get('batting', {})
        appearances = batting.get('plateAppearances')
        if appearances is None and str(player['id']) in confirmed_bench:
            appearances = 0
        observed = isinstance(appearances, (int, float)) and appearances >= 0
        played = observed and appearances > 0
        required = ['homeRuns', 'hits', 'runs', 'rbi', 'strikeOuts']
        complete = played and all(isinstance(batting.get(k), (int, float)) for k in required)
        outcomes.append({'id': player['id'], 'name': player.get('name'), 'team': player.get('team'),
                         'score': player.get('heat'), 'state': 'graded' if complete else 'no_recorded_appearance' if observed and not played else 'stats_unavailable',
                         'home_runs': batting['homeRuns'] if complete else None,
                         'hits': batting['hits'] if complete else None,
                         'hits_runs_rbis': sum(batting[k] for k in ['hits', 'runs', 'rbi']) if complete else None})
    return {'grader_version': 2, 'needs_retry': any(p['state'] == 'stats_unavailable' for p in outcomes),
            'game_pk': game_id, 'date': record['date'], 'captured_at': record['captured_at'],
            'game': record.get('game'), 'graded_at': datetime.now(timezone.utc).isoformat(),
            'source': f'https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore', 'players': outcomes}


def run(root='docs'):
    root = Path(root)
    output = root / 'pregame-results.json'
    previous = json.loads(output.read_text()) if output.exists() else {'games': []}
    graded = {str(g['game_pk']): g for g in previous.get('games', [])}
    pending, errors = 0, []
    session = requests.Session()
    for file in sorted((root / 'snapshots/pregame').glob('*/*.json')):
        record = json.loads(file.read_text())
        existing = graded.get(str(record['game_pk']), {})
        if existing.get('grader_version') == 2 and not existing.get('needs_retry'):
            continue
        try:
            response = session.get('https://statsapi.mlb.com/api/v1/schedule', params={'gamePk': record['game_pk']}, timeout=20)
            response.raise_for_status()
            schedule = response.json()
            games = [g for d in schedule.get('dates', []) for g in d.get('games', []) if g.get('gamePk') == record['game_pk']]
            if not games or games[0].get('status', {}).get('abstractGameState') != 'Final':
                pending += 1
                continue
            response = session.get(f"https://statsapi.mlb.com/api/v1/game/{record['game_pk']}/boxscore", timeout=20)
            response.raise_for_status()
            result = grade_record(record, schedule, response.json())
            if result:
                graded[str(record['game_pk'])] = result
                pending += int(result['needs_retry'])
        except (requests.RequestException, ValueError, KeyError) as error:
            pending += 1
            if existing:
                existing['needs_retry'] = True
            errors.append({'game_pk': record.get('game_pk'), 'reason': str(error)})
    payload = {'schema_version': 1, 'record_kind': 'pregame_model_research', 'updated_at': datetime.now(timezone.utc).isoformat(),
               'games': sorted(graded.values(), key=lambda g: (g['date'], g['game_pk'])), 'pending_games': pending, 'errors': errors,
               'note': 'Research scores recorded before first pitch. No frozen bet price: betting returns and prediction calibration are not measured here.'}
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    os.replace(temporary, output)
    print(f"Pregame research: {len(graded)} games graded, {pending} pending, {len(errors)} feed errors")


if __name__ == '__main__': run()
