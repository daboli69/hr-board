import json
import tempfile
import unittest
from pathlib import Path
from etl.signal_records import records_from_game, build

class SignalRecordTests(unittest.TestCase):
    def record(self):
        return dict(record_kind='pregame_model_research', captured_at='2026-09-21T15:00:00Z',generated_at='2026-09-21T14:00:00Z',kickoff='2026-09-21T22:00:00Z',game_pk=1,date='2026-09-21',game={'home':'BAL','away':'TOR'},players=[{'id':1,'name':'Player','heat':65,'sample':{'L15':40},'mix_punish':{'score':50}}])
    def test_no_leakage_or_fabricated_historical_features(self):
        r=self.record();row=records_from_game(r)[0]
        self.assertEqual(row['sample_sizes']['L15'],40)
        self.assertIsNone(row['features']['square_up'])
        r['captured_at']='2026-09-22T15:00:00Z'
        self.assertEqual(records_from_game(r),[])
    def test_first_features_immutable_and_real_outcome_attached(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'snapshots/pregame/2026-09-21';source.mkdir(parents=True)
            r=self.record();f=source/'1.json';f.write_text(json.dumps(r));build(root)
            r['players'][0]['heat']=99;f.write_text(json.dumps(r))
            (root/'pregame-results.json').write_text(json.dumps({'games':[{'game_pk':1,'graded_at':'2026-09-22T03:00:00Z','source':'official','players':[{'id':1,'state':'graded','home_runs':0,'hits':2,'hits_runs_rbis':3}]}]}))
            row=build(root)['records'][0]
            self.assertEqual(row['strength'],65)
            self.assertEqual(row['outcome']['values']['hits'],2)
