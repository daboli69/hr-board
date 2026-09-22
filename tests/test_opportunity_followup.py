import copy
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch
from etl.fetch_odds import refresh_sources, main as refresh_main
from etl.opportunity_data import save
from etl.build_opportunities import retain_failed_sources, freeze
from etl.grade_opportunities import settle
from etl.opportunity_models import valid_frozen_price, pitching_features, pitcher_projection
from etl.opportunity_followup import game_baseline, roi_summary


def offer(kind='game_line',identifier='game'):
    return {'id':identifier,'opportunity_type':kind,'freshness':'FRESH','rank':1,
            'observed_at':'2026-09-22T10:00:00Z','price':{'quoted_at':'2026-09-22T10:00:00Z','decimal':2.},
            'event':{'id':'1','start':'2026-09-22T23:00:00Z'},'entity':{'id':'game'},
            'market':'moneyline','side':'home','market_line':None,'split_overlap':{},
            'sample_sizes':{},'devigged_edge':.04,'source':{},'going_probability':.54}


class Followup(unittest.TestCase):
    def test_three_actual_refresh_entrypoint_runs_persist_stale_native_prices(self):
        error=urllib.error.HTTPError('https://provider.invalid/odds',502,'Bad Gateway',{},None)
        rows=[{'player':'Example','market_key':'player_home_runs','line':.5,'over_price':400,
               'bookmaker':'fanduel','game_date':'2026-09-22'}]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); odds=root/'odds.json'; raw=root/'opportunity_markets.json'
            save(raw,{'games':{'status':'FRESH','updated_at':'2026-09-22T10:00:00Z','rows':[{'price':2.}]}})
            with patch.dict('os.environ',{'PARLAY_API_KEY':'not-a-real-key'}),patch('etl.fetch_odds.OUT_PATH',odds),\
                 patch('etl.fetch_odds.fetch_props',return_value=rows),patch('etl.fetch_odds.fetch_game_lines',side_effect=error),\
                 patch('etl.fetch_odds._apply_movement'),patch('etl.fetch_odds._write',side_effect=lambda p:save(odds,p)):
                for _ in range(3):
                    self.assertEqual(refresh_main(),1)
                    saved=json.loads(raw.read_text())
                    self.assertEqual(saved['games']['status'],'STALE')
                    self.assertEqual(saved['games']['updated_at'],'2026-09-22T10:00:00Z')
                    self.assertEqual(saved['games']['rows'],[{'price':2.}])
                    self.assertEqual(saved['props']['status'],'FRESH')
                    self.assertFalse(json.loads(odds.read_text())['stale'])

    def test_roi_uses_real_price_and_one_simultaneous_book(self):
        p=offer(); p['expected_return']=.08
        r={'features':{'prediction':p},'outcome':{'values':{'status':'win'}}}
        p['devigged_market_probability']=.5
        lower=copy.deepcopy(r); lower['features']['prediction']['price']['decimal']=1.9
        summary=roi_summary([r,lower])['families']['game_lines']
        self.assertEqual(summary['settled_bets'],1)
        self.assertEqual(summary['roi_pct'],100)
        self.assertEqual(summary['average_realized_edge'],.5)

    def test_roi_empty_is_undefined_not_zero_and_pending_is_not_loss(self):
        p=offer(); p.update(expected_return=.08,devigged_market_probability=.5)
        r={'features':{'prediction':p},'outcome':None}
        summary=roi_summary([r])['families']['game_lines']
        self.assertEqual(summary['pending'],1)
        self.assertIsNone(summary['roi_pct'])
        self.assertIsNone(summary['win_rate_excluding_pushes'])

    def test_roi_never_chooses_a_better_later_price(self):
        p=offer(); p.update(expected_return=.08,devigged_market_probability=.5)
        r={'features':{'prediction':p},'outcome':{'values':{'status':'loss'}}}
        later=copy.deepcopy(r); later['features']['prediction'].update(observed_at='2026-09-22T12:00:00Z')
        later['features']['prediction']['price']['decimal']=5
        summary=roi_summary([r,later])['families']['game_lines']
        self.assertEqual(summary['settled_bets'],1)
        self.assertEqual(summary['roi_pct'],-100)

    def test_baseline_symmetry(self):
        p=game_baseline([],'A','B','2026-09-22')
        self.assertEqual(p['moneyline'],.5)
        self.assertTrue(0<p['spread']<.5)
        self.assertTrue(0<p['total']<1)

    def test_k_candidate_changes_only_short_workload_variance(self):
        f=pitching_features([{'date':'2026-09-01','started':True,'bf':8,'ip':2,'k':2,'runs':1},
                            {'date':'2026-09-05','started':True,'bf':20,'ip':4,'k':4,'runs':2}],'2026-09-22')
        a=pitcher_projection(f,[{'pa':300,'k':70}]); b=pitcher_projection(f,[{'pa':300,'k':70}],short_start_smoothing=True)
        self.assertEqual(a['projection'],b['projection'])
        self.assertEqual(a['features'],b['features'])
        self.assertNotEqual(a['distribution'],b['distribution'])
        f['expected_ip']=5
        self.assertEqual(pitcher_projection(f,[]),pitcher_projection(f,[],short_start_smoothing=True))

    def test_three_game_outages_preserve_prices_and_props_then_recover(self):
        original='2026-09-22T10:00:00Z'
        sources={key:{'status':'FRESH','updated_at':original,'rows':[{'price':2.}]} for key in ('props','games')}
        previous={'opportunities':[offer()], 'status':'FRESH'}
        error=urllib.error.HTTPError('https://provider.invalid/odds',502,'Bad Gateway',{},None)
        for hour in (12,14,16):
            now=f'2026-09-22T{hour}:00:00Z'
            with patch('etl.fetch_odds.fetch_props',return_value=[{'price':1.9}]),patch('etl.fetch_odds.fetch_game_lines',side_effect=error):
                sources=refresh_sources('not-a-real-key',sources,now)
            self.assertEqual(sources['props']['status'],'FRESH')
            self.assertEqual(sources['games']['status'],'STALE')
            self.assertEqual(sources['games']['updated_at'],original)
            fresh=offer('player_prop','fresh-prop'); fresh['observed_at']=now; fresh['price']['quoted_at']=now
            result={'generated_at':now,'status':'FALLBACK','opportunities':[fresh]}
            retain_failed_sources(result,previous,sources)
            stale=next(r for r in result['opportunities'] if r['id']=='game')
            self.assertEqual(stale['price'],offer()['price'])
            self.assertEqual(stale['observed_at'],original)
            self.assertEqual(stale['freshness'],'STALE')
            self.assertEqual(stale['stale_reason'],'http_502')
            self.assertEqual(stale['quote_age_seconds'],(hour-10)*3600)
            with tempfile.TemporaryDirectory() as temp:
                freeze(result,temp)
                records=json.loads((Path(temp)/'opportunity_predictions.json').read_text())['records']
                self.assertEqual([r['id'] for r in records],['fresh-prop'])
            previous=copy.deepcopy(result)
        with patch('etl.fetch_odds.fetch_props',return_value=[]),patch('etl.fetch_odds.fetch_game_lines',return_value=[{'price':2.1}]):
            recovered=refresh_sources('not-a-real-key',sources,'2026-09-22T18:00:00Z')
        self.assertEqual(recovered['games']['status'],'FRESH')
        self.assertEqual(recovered['games']['rows'][0]['price'],2.1)

    def test_props_failure_does_not_prevent_game_fetch(self):
        error=urllib.error.HTTPError('https://provider.invalid/props',502,'Bad Gateway',{},None)
        prior={'props':{'status':'FRESH','updated_at':'2026-09-22T10:00:00Z','rows':[{'price':2.}]}}
        with patch('etl.fetch_odds.fetch_props',side_effect=error),patch('etl.fetch_odds.fetch_game_lines',return_value=[{'new':True}]) as games:
            result=refresh_sources('not-a-real-key',prior,'2026-09-22T12:00:00Z')
        games.assert_called_once()
        self.assertEqual(result['props']['status'],'STALE')
        self.assertEqual(result['games']['status'],'FRESH')

    def test_cold_start_down_is_failed_not_stale(self):
        with patch('etl.fetch_odds.fetch_props',side_effect=TimeoutError),patch('etl.fetch_odds.fetch_game_lines',side_effect=TimeoutError):
            result=refresh_sources('not-a-real-key',{},'2026-09-22T12:00:00Z')
        self.assertEqual(result['games']['status'],'FAILED')
        self.assertEqual(result['props']['status'],'FAILED')

    def test_stale_capture_deferred_but_valid_frozen_bet_settles(self):
        final={'status':{'abstractGameState':'Final'},'teams':{'home':{'score':4},'away':{'score':2}}}
        p=offer()
        self.assertEqual(settle(p,final,{})['status'],'win')
        p['freshness']='STALE'
        self.assertIsNone(settle(p,final,{}))
        p=offer(); p['price']['quoted_at']='2026-09-21T10:00:00Z'
        self.assertFalse(valid_frozen_price(p))
        self.assertIsNone(settle(p,final,{}))
        p=offer(); p['price']['quoted_at']=p['event']['start']
        self.assertFalse(valid_frozen_price(p))


if __name__=='__main__': unittest.main()
