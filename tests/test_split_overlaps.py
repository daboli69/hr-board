import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from etl.split_overlaps import build, split_stats, recent_mix, rank_rows
from etl.pregame_records import freeze_games
from etl.grade_pregame import grade_record, run as grade_run


def log(date='2026-08-01', home=True, pk=10, **stats):
    return {'date': date, 'isHome': home, 'gameType': 'R', 'game': {'gamePk': pk}, 'stat':
            dict(hits=2, atBats=4, doubles=1, triples=0, homeRuns=1, baseOnBalls=1,
                 hitByPitch=0, plateAppearances=5, battersFaced=5, strikeOuts=1, sacFlies=0,
                 inningsPitched='1.2', runs=2, rbi=3, earnedRuns=1, gamesStarted=1, **stats)}


def pitches():
    return pd.DataFrame([dict(game_date='2026-08-01', game_pk=10, batter=1,pitcher=2,
                             inning_topbot=half,type='X',bb_type='fly_ball',launch_speed_angle=6,
                             launch_speed=100,events='home_run',description='hit_into_play',
                             woba_value=2,woba_denom=1,stand=hand,pitch_type=pitch)
                         for half, hand,pitch in [('Bot','R','FF'),('Top','L','SL')]])


def board():
    return {'slate_date':'2026-08-02','generated_at':'2026-08-02T12:00:00Z',
            'players':[{'id':1,'name':'Hitter','game_pk':20,'team':'BOS','opp_team':'NYY',
                        'side':'home','lineup_spot':2,'lineup_status':'confirmed',
                        'opp_pitcher':{'id':2,'name':'Starter'}}]}


class SplitTests(unittest.TestCase):
    def test_only_applicable_venue_and_prior_post_break_games(self):
        rows=[log(),log(home=False),log(date='2026-07-01'),log(date='2026-08-02')]
        result=split_stats(rows,pitches(),True,'2026-07-15','2026-08-02')
        self.assertEqual(result['pa'],5)
        self.assertEqual(result['rbi'],3)
        self.assertEqual(result['avg'],.5)
        self.assertEqual(result['iso'],1)
        self.assertEqual(result['bbe'],1)
        self.assertEqual(result['contact_pct'],100)

    def test_pitching_outs_and_home_half(self):
        result=split_stats([log()],pitches(),True,'2026-07-15','2026-08-02',True)
        self.assertAlmostEqual(result['ip'],5/3,places=4)
        self.assertEqual(result['whip'],1.8)
        self.assertEqual(result['hr9'],5.4)
        self.assertEqual(result['bbe'],1)

    def test_five_actual_starts_and_handedness(self):
        logs=[log(date=f'2026-07-{day:02}',pk=day) for day in range(20,27)]
        logs.append({'date':'2026-08-01','gameType':'R','game':{'gamePk':10},'stat':{'gamesStarted':0}})
        mix=recent_mix(pitches(),logs,'2026-08-02')
        self.assertEqual([s['game_pk'] for s in mix['starts']],[26,25,24,23,22])
        mix=recent_mix(pitches(),[log()],'2026-08-02')
        self.assertEqual(mix['starts_count'],1)
        self.assertEqual(mix['rows'][0]['usage_pct'],50)
        self.assertEqual(mix['rows'][0]['usage_pct_vs_R'],100)

    def test_no_other_signals_and_no_today_leakage(self):
        b=board(); frame=pitches()
        one=build(b,frame,'2026-07-15',{1:[log()]},{2:[log(home=False)]})
        b['players'][0].update(heat=999,park_hr_factor=7,odds=100,weather={'temp':120})
        extra=frame.copy();extra.game_date='2026-08-02'
        two=build(b,pd.concat([frame,extra]),'2026-07-15',{1:[log()]},{2:[log(home=False)]})
        self.assertEqual(one,two)
        self.assertEqual(len(one['boards']),3)
        self.assertIsNone(one['boards']['HR_OVERLAP'][0]['actual_outcome'])

    def test_threshold_independence_and_full_field(self):
        result=build(board(),pitches(),'2026-07-15',{1:[log()]},{2:[log(home=False)]})
        rows=result['boards']['HR_OVERLAP'];score=rows[0]['overlap_strength']
        self.assertEqual(len(rank_rows(rows,{'pa':5,'bf':5})),1)
        self.assertEqual(rank_rows(rows,{'pa':6,'bf':5}),[])
        self.assertEqual(rows[0]['overlap_strength'],score)

    def test_missing_pitch_data_is_not_zero_or_ranked(self):
        result=build(board(),pitches().iloc[0:0],'2026-07-15',{1:[log()]},{2:[log(home=False)]})
        self.assertIsNone(result['boards']['HR_OVERLAP'][0]['overlap_strength'])

    def test_doubleheader_rows_remain_distinct(self):
        b=board();b['players'].append(dict(b['players'][0],game_pk=21))
        result=build(b,pitches(),'2026-07-15',{1:[log()]},{2:[log(home=False)]})
        self.assertEqual([r['game_pk'] for r in result['boards']['HR_OVERLAP']],[20,21])

    def test_frozen_features_thresholds_and_outcomes(self):
        b=board();data=build(b,pitches(),'2026-07-15',{1:[log()]},{2:[log(home=False)]})
        snap={'date':b['slate_date'],'players':b['players'],'split_overlaps':data}
        games=[{'game_pk':20,'time':'2026-08-02T18:00:00Z'}]
        with tempfile.TemporaryDirectory() as root:
            freeze_games(snap,games,root,b['generated_at'],datetime(2026,8,2,13,tzinfo=timezone.utc))
            path=Path(root)/'pregame/2026-08-02/20.splits.json';original=path.read_bytes()
            data['boards']['HR_OVERLAP'][0]['overlap_strength']=999
            freeze_games(snap,games,root,b['generated_at'],datetime(2026,8,2,14,tzinfo=timezone.utc))
            self.assertEqual(original,path.read_bytes())
            record=json.loads(original)
            batting={'plateAppearances':4,'homeRuns':1,'hits':2,'runs':1,'rbi':3,'strikeOuts':0}
            team={'teamStats':{'batting':{'plateAppearances':4}},'players':{'ID1':{'person':{'id':1},'stats':{'batting':batting}}}}
            schedule={'dates':[{'games':[{'gamePk':20,'status':{'abstractGameState':'Final'}}]}]}
            result=grade_record(record,schedule,{'teams':{'home':team,'away':team}})
            self.assertEqual([r['actual_outcome'] for r in result['split_signals']],[1,2,6])
            self.assertEqual(result['split_signals'][0]['thresholds'],{'pa':50,'bf':30})

    def test_existing_baseline_does_not_prevent_first_split_record(self):
        b=board();games=[{'game_pk':20,'time':'2026-08-02T18:00:00Z'}]
        snap={'date':b['slate_date'],'players':b['players']}
        now=datetime(2026,8,2,13,tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as root:
            freeze_games(snap,games,root,b['generated_at'],now)
            path=Path(root)/'pregame/2026-08-02/20.json';original=path.read_bytes()
            snap['split_overlaps']=build(b,pitches(),'2026-07-15',{1:[log()]},{2:[log(home=False)]})
            freeze_games(snap,games,root,b['generated_at'],now)
            self.assertEqual(original,path.read_bytes())
            self.assertTrue(path.with_name('20.splits.json').exists())

    def test_grader_keeps_split_results_separate_from_baseline(self):
        b=board();snap={'date':b['slate_date'],'players':b['players'],
            'split_overlaps':build(b,pitches(),'2026-07-15',{1:[log()]},{2:[log(home=False)]})}
        games=[{'game_pk':20,'time':'2026-08-02T18:00:00Z'}]
        batting={'plateAppearances':4,'homeRuns':0,'hits':1,'runs':2,'rbi':1,'strikeOuts':1}
        team={'teamStats':{'batting':{'plateAppearances':4}},'players':{'ID1':{'person':{'id':1},'stats':{'batting':batting}}}}
        class Response:
            def __init__(self,data):self.data=data
            def raise_for_status(self):pass
            def json(self):return self.data
        class Session:
            def get(self,url,**kwargs):
                return Response({'teams':{'home':team,'away':team}} if 'boxscore' in url else
                    {'dates':[{'games':[{'gamePk':20,'status':{'abstractGameState':'Final'}}]}]})
        with tempfile.TemporaryDirectory() as root, patch('etl.grade_pregame.requests.Session',Session):
            freeze_games(snap,games,Path(root)/'snapshots',b['generated_at'],datetime(2026,8,2,13,tzinfo=timezone.utc))
            grade_run(root)
            result=json.loads((Path(root)/'pregame-results.json').read_text())
            self.assertEqual(len(result['games']),1)
            self.assertEqual(result['games'][0]['split_signals'],[])
            self.assertEqual([r['actual_outcome'] for r in result['split_games'][0]['split_signals']],[0,1,4])


if __name__=='__main__':unittest.main()
