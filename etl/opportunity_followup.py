"""Fixed v1 holdout, additional baselines, one K candidate, and honest ROI.

Log5 specifies a winner, not a total. The baseline uses v1's unadjusted
shrunk RS/RA rates and count dispersion to supply that missing score scale.
Winner strata are normalized to the exact original Pythagorean/log5 probability.
No market odds, starters, recent form or park inputs enter this baseline.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from etl.opportunity_data import MLB, final_scores, lineup_rates, save
from etl.opportunity_models import (team_rates, count_dist, pitching_features,
    pitcher_projection, game_projection, settlement_prob, utc, hr_probability, valid_frozen_price)
from etl.backtest_opportunities import metrics


def game_baseline(scores, home, away, cutoff):
    h,a=team_rates(scores,home,cutoff),team_rates(scores,away,cutoff)
    odds=(h['rs']/h['ra'])**1.83/(a['rs']/a['ra'])**1.83
    target=odds/(1+odds)
    hm,am=h['rs']*a['ra']/4.4,a['rs']*h['ra']/4.4
    hd,ad=count_dist(hm,1.5),count_dist(am,1.5)
    strata={True:[],False:[]}
    for hn,hp in hd.items():
        for an,ap in ad.items():
            mass=hp*ap
            if hn==an:
                strata[True].append((1,hn+an+1,mass*target))
                strata[False].append((-1,hn+an+1,mass*(1-target)))
            else: strata[hn>an].append((hn-an,hn+an,mass))
    margin,total=defaultdict(float),defaultdict(float)
    for winner,rows in strata.items():
        scale=(target if winner else 1-target)/sum(p for _,_,p in rows)
        for m,t,p in rows: margin[m]+=p*scale; total[t]+=p*scale
    return {'moneyline':target,'spread':settlement_prob(margin,1.5)['win'],
            'total':settlement_prob(total,8.5)['win']}


def cases_v1(root,api):
    original=json.loads((root/'opportunity_backtest.json').read_text())
    schedule=api.schedule('2026-03-01','2026-09-22')
    scores=final_scores(schedule); byid={s['game_pk']:s for s in scores}
    cases=[]; pitchers=set(); batters=set()
    for record in original['cases']:
        score=byid[record['game_pk']]; box=api.box(score['game_pk'])
        starters={}; lineups={}; actual={}
        for side in ('home','away'):
            players=box['teams'][side]['players'].values()
            sp=next(p for p in players if p.get('stats',{}).get('pitching',{}).get('gamesStarted')==1)
            starters[side]=sp['person']['id']; actual[side]=sp['stats']['pitching']['strikeOuts']
            lineups[side]=[p['person']['id'] for p in players if str(p.get('battingOrder','')).endswith('00')]
            pitchers.add(starters[side]); batters.update(lineups[side])
        cases.append({'score':score,'starters':starters,'lineups':lineups,'actual_k':actual})
    return scores,cases,api.logs(pitchers,2026),api.logs(batters,2026,'hitting')


def baseline_run(root=Path('docs')):
    api=MLB(); scores,cases,pitching,batting=cases_v1(root,api)
    rows=defaultdict(list); diagnostics=[]
    for c in cases:
        s=c['score']; cutoff=s['date']
        features={side:pitching_features(pitching[c['starters'][side]],cutoff) for side in ('home','away')}
        model=game_projection(scores,s['home'],s['away'],cutoff,features['home'],features['away'])
        base=game_baseline(scores,s['home'],s['away'],cutoff)
        outcomes={'moneyline':int(s['home_runs']>s['away_runs']),
                  'spread':int(s['home_runs']-s['away_runs']>1.5),'total':int(s['home_runs']+s['away_runs']>8.5)}
        probabilities={'moneyline':model['home_probability'],'spread':settlement_prob(model['margin_dist'],1.5)['win'],
                       'total':settlement_prob(model['total_dist'],8.5)['win']}
        for market,y in outcomes.items():
            rows[market].append({'p':probabilities[market],'y':y})
            rows[market+'_baseline'].append({'p':base[market],'y':y})
        for side,opp in [('home','away'),('away','home')]:
            f=features[side]
            if f['starts']<3: continue
            lineup=lineup_rates(c['lineups'][opp],batting,cutoff)
            k=pitcher_projection(f,lineup); y=int(c['actual_k'][side]>4.5)
            bp=settlement_prob(count_dist(f['season_k']*f['expected_bf']),4.5)['win']
            mp=settlement_prob(k['distribution'],4.5)['win']
            diagnostics.append({'game_pk':s['game_pk'],'pitcher':c['starters'][side],'features':f,'lineup':lineup,
                'actual_k':c['actual_k'][side],'model_probability':mp,'baseline_probability':bp,
                'opponent_k':k['features']['opposing_lineup_k'],
                'brier_loss_delta':(mp-y)**2-(bp-y)**2})
    report={'fixed_game_ids':[c['score']['game_pk'] for c in cases],
            'game_metrics':{k:metrics(v) for k,v in rows.items()},'k_diagnostics':diagnostics,
            'source_calls':api.calls}
    save(root/'opportunity_followup.json',report)
    print(json.dumps(report['game_metrics']))
    for label,test in [('short_workload',lambda r:r['features']['expected_ip']<4.5),
                       ('ordinary_workload',lambda r:r['features']['expected_ip']>=4.5),
                       ('lineup_below_league',lambda r:r['opponent_k']<.225),
                       ('lineup_above_league',lambda r:r['opponent_k']>=.225)]:
        group=[r for r in diagnostics if test(r)]
        print(label,{'n':len(group),'mean_loss_delta':sum(r['brier_loss_delta'] for r in group)/len(group) if group else None,
                     'mean_opponent_k':sum(r['opponent_k'] for r in group)/len(group) if group else None})


def k_attempt(root=Path('docs')):
    report=json.loads((root/'opportunity_followup.json').read_text())
    rows=defaultdict(list)
    for r in report['k_diagnostics']:
        candidate=pitcher_projection(r['features'],r['lineup'],short_start_smoothing=True)
        p=settlement_prob(candidate['distribution'],4.5)['win']; y=int(r['actual_k']>4.5)
        r['candidate_probability']=p
        for key,prob in [('v1',r['model_probability']),('candidate',p),('baseline',r['baseline_probability'])]:
            rows[key].append({'p':prob,'y':y})
    report['k_attempt']={'adjustment':'Remove workload mixture only for expected IP < 4.5; unchanged means, K rates and dispersion.',
        'reason':'Short-workload profiles accounted for disproportionate v1 excess Brier loss (0.0270 versus 0.00264 for ordinary starts).',
        'metrics':{k:metrics(v) for k,v in rows.items()},
        'qualification':'Same holdout used to diagnose and test this one adjustment: exploratory, not independent confirmation.'}
    save(root/'opportunity_followup.json',report)
    print(json.dumps(report['k_attempt']))


def extended_players(root):
    grades=json.loads((root/'pregame-results.json').read_text())
    actual={(g['game_pk'],p['id']):p for g in grades.get('games',[]) for p in g['players'] if p.get('state')=='graded'}
    days=defaultdict(list)
    for file in sorted((root/'snapshots/pregame').glob('*/*.json')):
        if '.splits.' in file.name: continue
        snap=json.loads(file.read_text())
        times=[utc(snap.get(k)) for k in ('generated_at','captured_at','kickoff')]
        if not all(times) or not times[0]<=times[1]<times[2]: continue
        for p in snap.get('players',[]):
            a=actual.get((snap['game_pk'],p['id']))
            if a: days[snap['date']].append((p,a))
    cal={}; rows=[]; count=0; hits=hrr=0
    for day,players in sorted(days.items()):
        for p,a in players:
            probability=hr_probability(p.get('heat'),cal)
            if probability is not None: rows.append({'p':probability,'y':int(a['home_runs']>0)})
            count+=1; hits+=a['hits']>0; hrr+=a['hits_runs_rbis']>1
        # Observations enter the existing interpolator only AFTER that day's test.
        for p,a in players:
            if p.get('heat') is None: continue
            key=str(max(0,min(90,int(p['heat']//10)*10)))
            e=cal.setdefault(key,{'n':0,'hr':0}); e['n']+=1; e['hr']+=a['home_runs']>0
    return {'all_frozen_graded_player_games':count,'date_start':min(days) if days else None,'date_end':max(days) if days else None,
            'hr_walk_forward':metrics(rows),'hits':{'n':count,'observed_rate':hits/count if count else None,'brier':None},
            'hrr':{'n':count,'observed_rate':hrr/count if count else None,'brier':None},
            'limitation':'Hit/HRR pregame probabilities absent from legacy frozen records. Outcomes are available, historical probability calibration is not.'}


def extended_run(root=Path('docs')):
    api=MLB(); schedule=api.schedule('2026-03-01','2026-09-22')
    scores=[s for s in final_scores(schedule) if s['date']<'2026-09-22']
    games={g['gamePk']:g for g in schedule}
    ids={g['teams'][s].get('probablePitcher',{}).get('id') for g in schedule for s in ('home','away')}
    batting_ids={p['id'] for g in schedule for s in ('home','away') for p in g.get('lineups',{}).get(s+'Players',[])}
    pitching=api.logs(ids,2026); batting=api.logs(batting_ids,2026,'hitting')
    # Actual-start game logs verify probable-starter identity; never grade a
    # scratched probable pitcher against another player's strikeout result.
    actual={}
    for pid,logs in pitching.items():
        for r in logs:
            if r.get('started') and isinstance(r.get('is_home'),bool):
                actual[(r['game_pk'],'home' if r['is_home'] else 'away')]=(pid,r)
    rows=defaultdict(list); missing=defaultdict(int); games_used=0
    fixed=set(json.loads((root/'opportunity_followup.json').read_text())['fixed_game_ids'])
    for score in scores:
        gid=score['game_pk']; cutoff=score['date']; g=games[gid]
        starters={s:actual.get((gid,s)) for s in ('home','away')}
        if not all(starters.values()): missing['actual_starter_unavailable']+=1; continue
        features={s:pitching_features(pitching[starters[s][0]],cutoff) for s in ('home','away')}
        model=game_projection(scores,score['home'],score['away'],cutoff,features['home'],features['away'])
        base=game_baseline(scores,score['home'],score['away'],cutoff)
        ps={'moneyline':model['home_probability'],'spread':settlement_prob(model['margin_dist'],1.5)['win'],
            'total':settlement_prob(model['total_dist'],8.5)['win']}
        ys={'moneyline':int(score['home_runs']>score['away_runs']),'spread':int(score['home_runs']-score['away_runs']>1.5),
            'total':int(score['home_runs']+score['away_runs']>8.5)}
        for key in ps:
            rows[key].append({'p':ps[key],'y':ys[key]}); rows[key+'_baseline'].append({'p':base[key],'y':ys[key]})
        games_used+=1
        for s,opp in [('home','away'),('away','home')]:
            f=features[s]
            if f['starts']<3: missing['k_less_than_three_prior_starts']+=1; continue
            lineup_ids=[p['id'] for p in g.get('lineups',{}).get(opp+'Players',[])]
            if len(lineup_ids)!=9: missing['k_missing_lineup']+=1; continue
            lineup=lineup_rates(lineup_ids,batting,cutoff)
            if any(not batting.get(pid) for pid in lineup_ids): missing['k_missing_batter_logs']+=1; continue
            k=pitcher_projection(f,lineup); candidate=pitcher_projection(f,lineup,short_start_smoothing=True)
            y=int(starters[s][1]['k']>4.5)
            for key,p in [('k',settlement_prob(k['distribution'],4.5)['win']),
                          ('k_candidate',settlement_prob(candidate['distribution'],4.5)['win']),
                          ('k_baseline',settlement_prob(count_dist(f['season_k']*f['expected_bf']),4.5)['win'])]:
                rows[key].append({'p':p,'y':y})
                if gid not in fixed: rows[key+'_outside_diagnostic_window'].append({'p':p,'y':y})
    report=json.loads((root/'opportunity_followup.json').read_text())
    report['extended']={'available_final_games':len(scores),'modeled_games':games_used,'start':scores[0]['date'],'end':scores[-1]['date'],
        'missing':dict(missing),'metrics':{k:metrics(v) for k,v in rows.items()},'player_results':extended_players(root),
        'source_calls':api.calls,'note':'Entire available 2026 regular season before Sept 22. Date-exclusive performance features; conditional on actual lineups/starters. Expanded set overlaps original diagnostic holdout.'}
    save(root/'opportunity_followup.json',report)
    print(json.dumps({'available':len(scores),'used':games_used,'missing':dict(missing),
        'metrics':{k:{f:v[f] for f in ('n','brier','ece')} for k,v in report['extended']['metrics'].items()},
        'players':report['extended']['player_results'],'source_calls':api.calls}))


def roi_summary(records, minimum_edge=.03):
    """Exploratory $1 stake, first observed offer, best simultaneous book only.

    Never join mutable daily odds onto an earlier feature holdout. Missing frozen
    prices/outcomes are missing observations, not assumed -110 prices or losses.
    """
    groups=defaultdict(list); excluded=defaultdict(int)
    for record in records:
        p=record.get('features',{}).get('prediction',{})
        if not valid_frozen_price(p): excluded['invalid_or_stale_capture']+=1; continue
        identity=(p['event']['id'],p['entity']['id'],p['market'],p.get('market_line'),p['side'])
        groups[identity].append(record)
    bets=defaultdict(list); candidates=defaultdict(int); pending=defaultdict(int)
    for identity,offers in groups.items():
        first=min(utc(r['features']['prediction']['observed_at']) for r in offers)
        simultaneous=[r for r in offers if utc(r['features']['prediction']['observed_at'])==first]
        record=max(simultaneous,key=lambda r:r['features']['prediction']['price']['decimal'])
        p=record['features']['prediction']; market=p['market']
        family='game_lines' if market in ('moneyline','spread','total') else market
        # Offered-price break-even is observable even for one-sided HR markets.
        # Never invent an Under price merely to make an ROI family eligible.
        win=p.get('going_probability'); push=p.get('push_probability',0)
        if win is None or push>=1: continue
        edge=win/(1-push)-1/p['price']['decimal']
        if edge<minimum_edge or (p.get('expected_return') or 0)<=0: continue
        candidates[family]+=1
        result=(record.get('outcome') or {}).get('values')
        if not result or result.get('status') not in ('win','loss','push','void'):
            pending[family]+=1; continue
        if result['status']=='void': excluded['void']+=1; continue
        won=result['status']=='win'; push=result['status']=='push'
        bets[family].append({'return':p['price']['decimal']-1 if won else 0 if push else -1,
            'won':won,'push':push,'forecast_edge':edge,
            'realized_edge':None if push else int(won)-1/p['price']['decimal']})
    result={}
    for family in ('game_lines','pk','hr','hits','hrr'):
        rows=bets[family]; decisive=[r for r in rows if not r['push']]
        result[family]={'qualifying_frozen_selections':candidates[family],'pending':pending[family],
            'settled_bets':len(rows),'stake_dollars':len(rows),'net_dollars':sum(r['return'] for r in rows) if rows else None,
            'win_rate_excluding_pushes':sum(r['won'] for r in decisive)/len(decisive) if decisive else None,
            'average_forecast_edge':sum(r['forecast_edge'] for r in rows)/len(rows) if rows else None,
            'average_realized_edge':sum(r['realized_edge'] for r in decisive)/len(decisive) if decisive else None,
            'roi_pct':100*sum(r['return'] for r in rows)/len(rows) if rows else None}
    return {'label':'EXPLORATORY — not future profitability','minimum_edge_vs_offered_break_even':minimum_edge,'minimum_expected_return':0,
            'stake_dollars':1,'selection_rule':'One bet per exact selection, first pregame observation; best book at that same observation time.',
            'realized_edge_definition':'Mean(actual win indicator minus 1/archived decimal price), excluding pushes; not a devigged-market comparison.',
            'families':result,'excluded':dict(excluded)}


def update_roi(root=Path('docs')):
    root=Path(root)
    if not (root/'opportunity_followup.json').exists(): return
    ledger=json.loads((root/'opportunity_predictions.json').read_text())
    report=json.loads((root/'opportunity_followup.json').read_text())
    report['roi']=roi_summary(ledger['records'])
    report['roi']['archive_start']=min((r['observed_at'][:10] for r in ledger['records']),default=None)
    report['roi']['historical_price_limitation']='Earlier 60-game/season reconstructions do not contain frozen offered prices. No retroactive price imputation; empty families have undefined ROI, not zero ROI.'
    save(root/'opportunity_followup.json',report)
    return report['roi']


def roi_run(root=Path('docs')):
    from etl.grade_opportunities import run as grade
    grade(root)
    print(json.dumps(update_roi(root)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['baselines','k','extended','roi'],default='baselines',nargs='?')
    args=parser.parse_args()
    {'baselines':baseline_run,'k':k_attempt,'extended':extended_run,'roi':roi_run}[args.stage]()
