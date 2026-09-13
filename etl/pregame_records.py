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


def freeze_games(snapshot, games, root, generated_at, now=None):
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
        # Exclusive creation preserves the original observation on later daily builds.
        try:
            with (folder / f'{game_id}.json').open('x', encoding='utf-8') as output:
                json.dump(record, output, default=str)
            written += 1
        except FileExistsError:
            pass
    return written
