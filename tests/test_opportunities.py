import unittest
from datetime import datetime, timezone
from tempfile import TemporaryDirectory
from pathlib import Path
import json
from unittest.mock import patch
from etl import fetch_odds
from etl.opportunity_models import (count_dist, settlement_prob, split_effect, pitching_features,
    pitcher_projection, game_projection, batter_projection, hr_probability)
from etl.build_opportunities import match_game, contract, freeze, build
from etl.grade_opportunities import settle, publish_ledger


class Models(unittest.TestCase):
    def test_fetches_one_complete_props_page(self):
        def fetch(url,key,retries):
            self.assertNotIn('apiKey',url)
            self.assertIn('player_hits_runs_rbis',url)
            fetch_odds.LAST_HEADERS.clear()
            fetch_odds.LAST_HEADERS.update({'x-result-has-more':'false'})
            return [{'player':'One'}]
        with patch.object(fetch_odds,'_fetch',side_effect=fetch) as call:
            self.assertEqual(len(fetch_odds.fetch_props('test')),1)
            self.assertEqual(call.call_count,1)

    def test_truncated_provider_data_is_not_complete(self):
        def fetch(*args):
            fetch_odds.LAST_HEADERS.clear()
            fetch_odds.LAST_HEADERS.update({'x-result-truncated':'true'})
            return [{'player':'One'}]
        with patch.object(fetch_odds,'_fetch',side_effect=fetch):
            with self.assertRaises(ValueError): fetch_odds.fetch_props('test')

    def test_props_pagination_is_bounded_and_shared(self):
        offsets=[]
        def fetch(url,*args):
            offset=int(url.rsplit('offset=',1)[1]); offsets.append(offset)
            fetch_odds.LAST_HEADERS.clear()
            fetch_odds.LAST_HEADERS.update({'x-result-has-more':'true','x-next-offset':str(offset+5000)})
            return [offset]
        with patch.object(fetch_odds,'_fetch',side_effect=fetch):
            with self.assertRaises(ValueError): fetch_odds.fetch_props('test')
        self.assertEqual(offsets,[0,5000,10000])

    def test_distribution_mass(self):
        for mean in (.01,1,5,12):
            for dispersion in (1,1.5,1.8):
                d=count_dist(mean,dispersion)
                self.assertAlmostEqual(sum(d.values()),1)
                self.assertAlmostEqual(sum(k*p for k,p in d.items()),mean,places=5)

    def test_push_and_devig(self):
        d=count_dist(4)
        o,u=settlement_prob(d,4),settlement_prob(d,4,'under')
        self.assertAlmostEqual(o['win']+u['win']+o['push'],1)
        self.assertAlmostEqual(o['conditional_win']+u['conditional_win'],1)

    def test_split_cap_and_neutral(self):
        for strength in (0,50,100,400):
            for conf in (0,10,50,100):
                e=split_effect({'overlap_strength':strength,'sample_confidence':conf})
                self.assertLessEqual(abs(e['relative_change']),.03)
                if strength==100 or conf==0: self.assertEqual(e['relative_change'],0)
        self.assertLess(abs(split_effect({'overlap_strength':400,'sample_confidence':10})['relative_change']),.001)

    def test_no_future_pitching(self):
        prior={'date':'2026-09-01','started':True,'bf':25,'k':7,'runs':2,'ip':6}
        future=dict(prior,date='2026-09-10',k=25,runs=99)
        self.assertEqual(pitching_features([prior],'2026-09-10'),pitching_features([prior,future],'2026-09-10'))

    def test_no_future_game(self):
        g={'home':'A','away':'B','date':'2026-09-10','home_runs':99,'away_runs':0}
        f=pitching_features([],'2026-09-10')
        a=game_projection([],'A','B','2026-09-10',f,f)
        b=game_projection([g],'A','B','2026-09-10',f,f)
        self.assertEqual(a,b)
        self.assertAlmostEqual(sum(a['margin_dist'].values()),1)
        self.assertAlmostEqual(sum(a['total_dist'].values()),1)
        self.assertNotIn(0,a['margin_dist'])
        self.assertNotIn('split_overlap',a['features'])

    def test_hit_reuse(self):
        p={'hit_gated':{'p_hit':.68,'xpa':4.55}}
        m=batter_projection(p,'hits',.5)
        self.assertAlmostEqual(m['probability'],.68,places=10)
        self.assertAlmostEqual(sum(m['distribution'].values()),1)
        self.assertLess(batter_projection(p,'hits',1.5)['probability'],.68)

    def test_hrr_reuse(self):
        p={'hrr_proj':{'hrr':1.97,'p_over15':.4194}}
        self.assertAlmostEqual(batter_projection(p,'hrr',1.5)['probability'],.4194,places=4)

    def test_hr_reuse(self):
        self.assertEqual(hr_probability(65,{'60':{'n':100,'hr':20},'70':{'n':100,'hr':30}}),.25)
        self.assertIsNone(hr_probability(65,{}))

    def test_pitcher_workload(self):
        f=pitching_features([{'date':'2026-09-01','started':True,'bf':10,'ip':2,'runs':1,'k':3}],'2026-09-10')
        m=pitcher_projection(f,[{'pa':400,'k':100}])
        self.assertLess(f['expected_ip'],5)
        self.assertAlmostEqual(sum(m['distribution'].values()),1)

    def test_doubleheader_identity(self):
        games=[{'game_pk':1,'home_name':'A','away_name':'B','time':'2026-09-10T17:00:00Z'},
               {'game_pk':2,'home_name':'A','away_name':'B','time':'2026-09-10T23:00:00Z'}]
        row={'home_team':'A','away_team':'B','commence_time':'2026-09-10T23:00:00Z'}
        self.assertEqual(match_game(row,games)['game_pk'],2)
        self.assertIsNone(match_game(dict(row,commence_time_reported=False),games))
        self.assertIsNone(match_game(dict(row,home_team='C'),games))

    def example(self):
        return contract({'game_pk':1,'home':'A','away':'B','time':'2026-09-10T23:00:00Z'},
            {'id':'10','name':'Player'},'hits',.5,'over',2.,2.,.6,1.,0.,'book','2026-09-10T20:00:00Z',
            {},{'batter_pa':400},['Real feature 1','Real feature 2'],'Role uncertainty',datetime(2026,9,10,20,tzinfo=timezone.utc))

    def test_contract(self):
        p=self.example()
        for k in ['opportunity_type','bet_description','market_line','price','going_projection','going_probability',
                  'devigged_market_probability','confidence_level','why','main_risk','parlay_fit']:
            self.assertIn(k,p)
        self.assertEqual(p['devigged_market_probability'],.5)
        self.assertAlmostEqual(p['devigged_edge'],.1)

    def test_immutable(self):
        with TemporaryDirectory() as d:
            p=self.example(); p['rank']=1
            payload={'opportunities':[p],'generated_at':p['observed_at']}
            freeze(payload,d); p['going_probability']=.99; freeze(payload,d)
            ledger=json.loads((Path(d)/'opportunity_predictions.json').read_text())
            self.assertEqual(ledger['records'][0]['features']['prediction']['going_probability'],.6)
            publish_ledger(d)
            self.assertEqual(len(json.loads((Path(d)/'signal_tracker.json').read_text())['records']),1)

    def test_settlement(self):
        p=self.example(); game={'status':{'abstractGameState':'Final'}}
        box={'teams':{'home':{'players':{'ID10':{'person':{'id':10},'stats':{'batting':{'plateAppearances':4,'hits':1}}}}}}}
        self.assertEqual(settle(p,game,box)['status'],'win')
        p['market_line']=1
        self.assertEqual(settle(p,game,box)['status'],'push')
        self.assertIsNone(settle(p,{'status':{'abstractGameState':'Live'}},box))


if __name__=='__main__': unittest.main()
