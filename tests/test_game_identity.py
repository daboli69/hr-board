"""Game-identity regression checks, without loading remote data providers."""
import ast
from pathlib import Path
import unittest
import pandas as pd

source = ast.parse(Path('etl/track.py').read_text(encoding='utf-8'))
keep = {'_game_map', '_bind_participants', '_selection_key', '_k_map'}
namespace = {'K_EVENTS': {'strikeout', 'strikeout_double_play'}}
exec(compile(ast.Module(body=[node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in keep], type_ignores=[]), '<tracking helpers>', 'exec'), namespace)

class GameIdentityTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([{'game_pk': 10, 'batter': 7, 'events': 'strikeout'}, {'game_pk': 11, 'batter': 7, 'events': 'single'}, {'game_pk': 11, 'batter': 8, 'events': 'strikeout'}])
    def test_games_never_combine(self):
        result = namespace['_game_map'](self.frame, namespace['_k_map'])
        self.assertEqual(result.get((10, 7)), 1)
        self.assertEqual(result.get((11, 7), 0), 0)
    def test_ambiguous_or_absent_players_are_excluded(self):
        result, excluded = namespace['_bind_participants']([{'id': 7}, {'id': 9}, {'id': 8}, {'id': 7, 'game_pk': 10}, {'id': 7, 'game_pk': 12}], self.frame)
        self.assertEqual(excluded, 3)
        self.assertEqual([(p['id'], p['game_pk']) for p in result], [(8, 11), (7, 10)])

if __name__ == '__main__': unittest.main()
