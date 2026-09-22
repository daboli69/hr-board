"""Chronological reconstruction against official finals; no fabricated closing lines.

Observed starter/lineup identities condition the retrospective test. Performance
features exclude the entire game date, including doubleheader game one. Archived
pregame park/weather inputs are not available here: report a neutral-environment
ablation explicitly, not a backtest of unavailable historical weather.
"""
import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from etl.opportunity_data import MLB, final_scores, lineup_rates, save
from etl.opportunity_models import (game_projection, pitching_features, pitcher_projection,
                                   team_rates, settlement_prob, count_dist, utc, hr_probability)


def metrics(rows):
    if not rows: return {'n':0}
    pairs=[(r['p'],r['y']) for r in rows]
    bins=[]
    for i in range(10):
        b=[(p,y) for p,y in pairs if i/10<=p<(i+1)/10 or i==9 and p==1]
        if b: bins.append({'n':len(b),'predicted':sum(p for p,y in b)/len(b),'observed':sum(y for p,y in b)/len(b)})
    losses=[(p-y)**2 for p,y in pairs]
    mean=sum(losses)/len(losses)
    se=(sum((x-mean)**2 for x in losses)/max(1,len(losses)-1)/len(losses))**.5
    return {'n':len(pairs),'brier':mean,'brier_approx_95ci':[max(0,mean-1.96*se),min(1,mean+1.96*se)],
            'log_loss':-sum(y*math.log(max(1e-9,p))+(1-y)*math.log(max(1e-9,1-p)) for p,y in pairs)/len(pairs),
            'ece':sum(b['n']*abs(b['predicted']-b['observed']) for b in bins)/len(pairs),
            'mean_probability':sum(p for p,y in pairs)/len(pairs),'observed_rate':sum(y for p,y in pairs)/len(pairs),
            'bins':bins}


def frozen_player_results(root):
    grades=json.loads((root/'pregame-results.json').read_text())
    actual={(g['game_pk'],p['id']):p for g in grades.get('games',[]) for p in g['players'] if p.get('state')=='graded'}
    rows=[]
    for file in sorted((root/'snapshots/pregame').glob('*/*.json')):
        if '.splits.' in file.name: continue
        snap=json.loads(file.read_text())
        if not utc(snap.get('generated_at')) or not utc(snap.get('captured_at')) or not utc(snap.get('kickoff')): continue
        if not utc(snap['generated_at'])<=utc(snap['captured_at'])<utc(snap['kickoff']): continue
        for p in snap.get('players',[]):
            a=actual.get((snap['game_pk'],p['id']))
            if a: rows.append((snap['date'],p,a))
    days=sorted({r[0] for r in rows}); cutoff=days[len(days)//2] if days else ''
    training=[r for r in rows if r[0]<cutoff]; testing=[r for r in rows if r[0]>=cutoff]
    cal={}
    for _,p,a in training:
        if p.get('heat') is None: continue
        b=str(max(0,min(90,int(p['heat']//10)*10)))
        e=cal.setdefault(b,{'n':0,'hr':0}); e['n']+=1; e['hr']+=a['home_runs']>0
    hr=[]
    for _,p,a in testing:
        prob=hr_probability(p.get('heat'),cal)
        if prob is not None: hr.append({'p':prob,'y':int(a['home_runs']>0)})
    return {'cutoff':cutoff,'training_player_games':len(training),'test_player_games':len(testing),
            'hr_chronological_existing_interpolator':metrics(hr),
            'hits':{'n':len(testing),'observed_over05':sum(a['hits']>0 for _,p,a in testing)/len(testing) if testing else None,
                    'calibration':'Unavailable: historical immutable records did not retain hit_gated probabilities.'},
            'hrr':{'n':len(testing),'observed_over15':sum(a['hits_runs_rbis']>1 for _,p,a in testing)/len(testing) if testing else None,
                   'calibration':'Unavailable: historical immutable records did not retain hrr_proj probabilities; existing fitted mean correction is not OOS proof.'},
            'note':'HR calibration trained only on earlier immutable records; hit/HRR outcome tracking reused without refitting.'}


def run(root='docs', n=60):
    root=Path(root); api=MLB(); today=date.today().isoformat(); year=today[:4]
    schedule=api.schedule(year+'-03-01',today)
    scores=final_scores(schedule)
    byid={g['gamePk']:g for g in schedule}
    tests=[g for g in scores if g['date']<today][-n:]
    boxes={}; starter_ids=set(); batter_ids=set(); cases=[]
    for score in tests:
        box=api.box(score['game_pk']); boxes[score['game_pk']]=box
        starters={}; lineups={}
        for side in ('home','away'):
            team=box['teams'][side]
            starters[side]=next((p['person']['id'] for p in team['players'].values() if p.get('stats',{}).get('pitching',{}).get('gamesStarted')==1),None)
            lineups[side]=[p['person']['id'] for p in team['players'].values() if str(p.get('battingOrder','')).endswith('00')]
            starter_ids.add(starters[side]); batter_ids.update(lineups[side])
        if all(starters.values()): cases.append((score,starters,lineups))
    pitching=api.logs(starter_ids,year); batting=api.logs(batter_ids,year,'hitting')
    measures=defaultdict(list); total_errors=[]; k_errors=[]; feature_cases=[]
    for score,starters,lineups in cases:
        cutoff=score['date']; hs=pitching_features(pitching.get(starters['home'],[]),cutoff); ass=pitching_features(pitching.get(starters['away'],[]),cutoff)
        model=game_projection(scores,score['home'],score['away'],cutoff,hs,ass)
        y=int(score['home_runs']>score['away_runs'])
        measures['moneyline'].append({'p':model['home_probability'],'y':y})
        # Published Pythagorean/log5 run-strength baseline, no current game data.
        h=team_rates(scores,score['home'],cutoff); a=team_rates(scores,score['away'],cutoff)
        odds=(h['rs']/h['ra'])**1.83/(a['rs']/a['ra'])**1.83
        measures['moneyline_baseline'].append({'p':odds/(1+odds),'y':y})
        total=score['home_runs']+score['away_runs']
        # Fixed test thresholds, NOT claimed historical sportsbook lines.
        measures['total_over85'].append({'p':settlement_prob(model['total_dist'],8.5)['win'],'y':int(total>8.5)})
        measures['spread_home_minus15'].append({'p':settlement_prob(model['margin_dist'],1.5)['win'],'y':int(score['home_runs']-score['away_runs']>1.5)})
        total_errors.append(abs(model['total']-total))
        for side,opp,features in [('home','away',hs),('away','home',ass)]:
            if features['starts']<3: continue
            km=pitcher_projection(features,lineup_rates(lineups[opp],batting,cutoff))
            actual=boxes[score['game_pk']]['teams'][side]['players']['ID'+str(starters[side])]['stats']['pitching']['strikeOuts']
            measures['pitcher_k_over45'].append({'p':settlement_prob(km['distribution'],4.5)['win'],'y':int(actual>4.5)})
            base=count_dist(features['season_k']*features['expected_bf'])
            measures['pitcher_k_baseline'].append({'p':settlement_prob(base,4.5)['win'],'y':int(actual>4.5)})
            k_errors.append(abs(km['projection']-actual))
        feature_cases.append({'game_pk':score['game_pk'],'cutoff_exclusive':cutoff,'home_probability':model['home_probability'],
                             'projected_total':model['total'],'actual_home':score['home_runs'],'actual_away':score['away_runs']})
    report={'schema_version':1,'generated_at':datetime.now(timezone.utc).isoformat(),'model_version':'yard-opportunities-1',
            'start':tests[0]['date'] if tests else None,'end':tests[-1]['date'] if tests else None,
            'metrics':{k:metrics(v) for k,v in measures.items()},
            'total_mae':sum(total_errors)/len(total_errors) if total_errors else None,
            'pitcher_k_mae':sum(k_errors)/len(k_errors) if k_errors else None,
            'player_tracking':frozen_player_results(root),'cases':feature_cases,'source_calls':api.calls,
            'limitations':['Conditional on actual starter and lineup identities; not reconstructed announcement timestamps.',
                'Neutral historical park/weather and no archived bullpen availability: core model validation only.',
                'Fixed 8.5 total, home -1.5 spread, K 4.5 diagnostics; not historical price/ROI backtests.',
                'Historical split-feature/probability joint snapshots unavailable; prospective with/without split probabilities now frozen.',
                'Small holdout; game-clustered uncertainty not established by per-prediction intervals.']}
    save(root/'opportunity_backtest.json',report)
    print(json.dumps({k:report[k] for k in ('metrics','total_mae','pitcher_k_mae','player_tracking','source_calls')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--games',type=int,default=60); args=p.parse_args(); run(n=args.games)
