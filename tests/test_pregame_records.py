from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
spec = importlib.util.spec_from_file_location('pregame', Path('etl/pregame_records.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class PregameTests(unittest.TestCase):
    def test_first_observation_is_frozen_and_started_games_are_excluded(self):
        now = datetime(2026, 9, 12, 18, tzinfo=timezone.utc)
        snap = {'date': '2026-09-12', 'players': [{'id': 7, 'game_pk': 10, 'heat': 60}, {'id': 7, 'game_pk': 11, 'heat': 80}]}
        games = [{'game_pk': 10, 'time': '2026-09-12T17:00:00Z'}, {'game_pk': 11, 'time': '2026-09-12T20:00:00Z'}]
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(module.freeze_games(snap, games, root, now.isoformat(), now), 1)
            path = Path(root) / 'pregame/2026-09-12/11.json'
            original = path.read_bytes()
            snap['players'][1]['heat'] = 99
            self.assertEqual(module.freeze_games(snap, games, root, now.isoformat(), now), 0)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(json.loads(original)['players'][0]['heat'], 80)
            self.assertFalse(path.with_name('10.json').exists())
    def test_missing_and_future_generation_times_cannot_create_evidence(self):
        now = datetime(2026, 9, 12, 18, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as root:
            for generated in [None, '2026-09-12T19:00:00Z', '2026-09-12T17:00:00']:
                self.assertEqual(module.freeze_games({'date':'2026-09-12'}, [], root, generated, now), 0)

if __name__ == '__main__': unittest.main()
