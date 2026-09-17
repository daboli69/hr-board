"""Keep untimestamped live season aggregates out of historical replay paths."""
import ast
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReplayTimePolicyTests(unittest.TestCase):
    def test_replay_cannot_fetch_current_fangraphs_aggregates(self):
        tree = ast.parse((ROOT / 'etl/backtest.py').read_text(encoding='utf-8'))
        calls = [node.func.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertNotIn('fangraphs_pitching', calls)
        for name in ('replay', 'replay_runs'):
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            entries = [(key.value, value.value) for n in ast.walk(function) if isinstance(n, ast.Dict)
                       for key, value in zip(n.keys, n.values)
                       if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)]
            self.assertIn(('pregame_validated', False), entries)

    def test_legacy_artifact_does_not_claim_corrected_rerun(self):
        data = json.loads((ROOT / 'docs/backtest.json').read_text(encoding='utf-8'))
        if data.get('validation_scope') == 'legacy_diagnostic_replay':
            self.assertTrue(data['rerun_required'])
        self.assertIs(data['pregame_validated'], False)


if __name__ == '__main__':
    unittest.main()
