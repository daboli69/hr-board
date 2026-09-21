"""Sport-neutral, immutable pregame feature ledger. No scores are refitted here."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from etl.pregame_records import _utc


def records_from_game(record):
    captured, generated, start = [_utc(record.get(k)) for k in ('captured_at', 'generated_at', 'kickoff')]
    if not captured or not generated or not start or not generated <= captured < start:
        return []
    if record.get('record_kind') != 'pregame_model_research':
        return []
    players = record.get('players', [])
    ranked = sorted((p for p in players if isinstance(p.get('heat'), (int, float))), key=lambda p: (-p['heat'], str(p['id'])))
    ranks = {p['id']: i + 1 for i, p in enumerate(ranked)}
    game = record.get('game') or {}
    rows = []
    for p in players:
        features = {k: p.get(k) for k in ('heat', 'hit_heat', 'hrr_heat', 'sp_vuln', 'vuln', 'sq_up', 'square_up', 'mix_punish', 'badges', 'park_boost', 'park_hr_factor', 'signals', 'metrics', 'spot', 'grade')}
        rows.append({'schema_version': 1, 'id': hashlib.sha256(f"mlb|{record['game_pk']}|{p['id']}|yard_signals".encode()).hexdigest(),
                     'sport': 'mlb', 'event': {'id': str(record['game_pk']), 'start': record['kickoff'], 'home': game.get('home'), 'away': game.get('away')},
                     'entity': {'id': str(p['id']), 'name': p.get('name')}, 'signal_family': 'YARD_SIGNALS',
                     'features': features, 'sample_sizes': p.get('sample') or {}, 'rank': ranks.get(p['id']),
                     'rank_scope': 'game', 'strength': p.get('heat'), 'model_version': record.get('model_version') or f"yard-window-{record.get('window_v', 'unknown')}",
                     'observed_at': record['captured_at'], 'source': {'generated_at': record['generated_at'], 'path': f"snapshots/pregame/{record['date']}/{record['game_pk']}.json"},
                     'outcome': None})
    return rows


def build(root='docs'):
    root = Path(root)
    output = root / 'signal_tracker.json'
    old = json.loads(output.read_text(encoding='utf-8')) if output.exists() else {}
    records = {r['id']: r for r in old.get('records', [])}
    for file in sorted((root / 'snapshots/pregame').glob('*/*.json')):
        for row in records_from_game(json.loads(file.read_text(encoding='utf-8'))):
            records.setdefault(row['id'], row)
    outcomes = json.loads((root / 'pregame-results.json').read_text(encoding='utf-8')) if (root / 'pregame-results.json').exists() else {}
    games = {str(g['game_pk']): g for g in outcomes.get('games', [])}
    for row in records.values():
        if row.get('outcome'):
            continue
        game = games.get(row['event']['id'], {})
        player = next((p for p in game.get('players', []) if str(p['id']) == row['entity']['id'] and p.get('state') == 'graded'), None)
        # The existing official-boxscore grader validates completion and appearances.
        if player:
            row['outcome'] = {'values': {k: player[k] for k in ('home_runs', 'hits', 'hits_runs_rbis')}, 'observed_at': game['graded_at'], 'source': game['source']}
    payload = {'schema_version': 1, 'generated_at': datetime.now(timezone.utc).isoformat(),
               'note': 'Immutable pregame research features. Unknown historical features remain null; no backfilled estimates or ROI claims.',
               'records': sorted(records.values(), key=lambda r: (r['observed_at'], r['id']))}
    temp = output.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    temp.replace(output)
    print(f"Signal ledger: {len(records)} pregame observations")
    return payload


if __name__ == '__main__':
    build()
