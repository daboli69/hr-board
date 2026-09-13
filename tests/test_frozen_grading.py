import ast
from datetime import datetime, timezone
from pathlib import Path
import unittest
module = ast.parse(Path('etl/grade_pregame.py').read_text())
def utc(value): return datetime.fromisoformat(value.replace('Z','+00:00'))
ns = {'_utc':utc,'datetime':datetime,'timezone':timezone}
exec(compile(ast.Module(body=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='grade_record'],type_ignores=[]),'<grader>','exec'),ns)
class FrozenGradingTests(unittest.TestCase):
 def test_official_final_required_and_missing_appearance_is_not_loss(self):
  record={'game_pk':1,'date':'2026-09-12','captured_at':'2026-09-12T16:00:00Z','generated_at':'2026-09-12T15:59:00Z','kickoff':'2026-09-12T17:00:00Z','players':[{'id':7},{'id':8}]}
  schedule={'dates':[{'games':[{'gamePk':1,'status':{'abstractGameState':'Live'}}]}]}
  box={'teams':{'home':{'players':{'ID7':{'person':{'id':7},'stats':{'batting':{'plateAppearances':4,'hits':2,'runs':1,'rbi':1,'homeRuns':1,'strikeOuts':0}}}}}}}
  self.assertIsNone(ns['grade_record'](record,schedule,box))
  schedule['dates'][0]['games'][0]['status']['abstractGameState']='Final'
  result=ns['grade_record'](record,schedule,box)
  self.assertEqual(result['players'][0]['hits_runs_rbis'],4)
  self.assertEqual(result['players'][1]['state'],'no_recorded_appearance')
  self.assertIsNone(result['players'][1]['home_runs'])
  record['captured_at']='2026-09-12T18:00:00Z'
  with self.assertRaises(ValueError):ns['grade_record'](record,schedule,box)
if __name__=='__main__':unittest.main()
