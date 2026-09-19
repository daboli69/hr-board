"""Write immutable pregame research records. Historical daily files remain legacy data."""
from datetime import datetime, timezone
import json
from pathlib import Path


def _utc(value):
    try:
        dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return dt.astimezone(timezone.utc) if dt.tzinfo else None
    except (ValueError, TypeError):
        return None


def freeze_games(snapshot, games, root, generated_at, now=None, split_only=False):
    """Freeze the first observed pregame feature set; never backfill started games.

    These are model research observations, not sportsbook tickets or placed bets.
    """
    now = now or datetime.now(timezone.utc)
    generated = _utc(generated_at)
    if generated is None or generated > now:
        return 0
    folder = Path(root) / 'pregame' / str(snapshot['date'])
    folder.mkdir(parents=True, exist_ok=True)
    written = 0
    for game in games:
        start = _utc(game.get('time'))
        game_id = game.get('game_pk')
        if start is None or start <= now or game_id is None:
            continue
        try:
            game_id = int(game_id)
        except (TypeError, ValueError):
            continue
        players = [p for p in snapshot.get('players', []) if str(p.get('game_pk')) == str(game_id)]
        # Some legacy board maps collapse doubleheaders by player ID. Preserve
        # split-only game/player identities so their outcomes can still be graded.
        present = {p['id'] for p in players}
        for row in ((snapshot.get('split_overlaps') or {}).get('boards') or {}).get('HR_OVERLAP', []):
            if row['game_pk'] == game_id and row['id'] not in present:
                players.append({'id': row['id'], 'name': row['name'], 'game_pk': game_id})
                present.add(row['id'])
        if not players:
            continue
        record = {
            'schema_version': 1, 'record_kind': 'pregame_model_research',
            'captured_at': now.isoformat(), 'generated_at': generated.isoformat(),
            'kickoff': start.isoformat(), 'date': snapshot['date'], 'game_pk': game_id,
            'game': game, 'players': players,
            'pitcher_props': [p for p in snapshot.get('pitcher_props', []) if str(p.get('game_pk')) == str(game_id)],
            'window_v': snapshot.get('window_v'),
        }
        split = snapshot.get('split_overlaps') or {}
        if split.get('status') == 'FRESH' and any(r['game_pk'] == game_id for r in split['boards'].get('HR_OVERLAP', [])):
            from etl.split_overlaps import rank_rows
            thresholds = split['default_thresholds']
            record['split_overlaps'] = {k: v for k, v in split.items() if k not in ('boards', 'rankings')}
            record['split_overlaps']['boards'] = {}
            for family, rows in split['boards'].items():
                ranked = {(r['game_pk'], r['id']): r['rank'] for r in rank_rows(rows, thresholds)}
                record['split_overlaps']['boards'][family] = [dict(r, rank=ranked.get((r['game_pk'], r['id'])),
                    thresholds=dict(thresholds)) for r in rows if r['game_pk'] == game_id]
            # A prior baseline record (or a failed early split fetch) must not
            # prevent the first successful pregame split observation being saved.
            split_record = dict(record, record_kind='split_overlap_research')
            try:
                with (folder / f'{game_id}.splits.json').open('x', encoding='utf-8') as output:
                    json.dump(split_record, output, default=str)
                if split_only:
                    written += 1
            except FileExistsError:
                pass
            record.pop('split_overlaps')
        if split_only:
            continue
        # Exclusive creation preserves the original observation on later daily builds.
        try:
            with (folder / f'{game_id}.json').open('x', encoding='utf-8') as output:
                json.dump(record, output, default=str)
            written += 1
        except FileExistsError:
            pass
    return written
