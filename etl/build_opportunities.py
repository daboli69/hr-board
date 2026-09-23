"""Produce all MLB opportunity families from shared static inputs and official MLB.

Run after build_board and fetch_odds. No secret or bookmaker fetch in this module.
"""
import argparse
import json
from zoneinfo import ZoneInfo
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from etl.fetch_odds import _norm_name, _prop_price_to_decimal, decimal_to_american
from etl.opportunity_data import MLB, final_scores, lineup_rates, save
from etl.opportunity_models import (VERSION, utc, pitching_features, game_projection,
                                   pitcher_projection, batter_projection, settlement_prob)

MARKETS={'player_hits':'hits','player_hits_runs_rbis':'hrr','player_home_runs':'hr',
         'player_strikeouts':'pk','player_pitcher_strikeouts':'pk'}


def match_game(row, games):
    """Names PLUS reported first-pitch time; never name-only doubleheader joins."""
    stamp=utc(row.get('commence_time'))
    if not stamp or row.get('commence_time_reported') is False: return None
    matches=[g for g in games if _norm_name(g['home_name'])==_norm_name(row.get('home_team',''))
             and _norm_name(g['away_name'])==_norm_name(row.get('away_team',''))
             and abs((utc(g['time'])-stamp).total_seconds())<=1800]
    return matches[0] if len(matches)==1 else None


def quote_fresh(timestamp, now):
    stamp=utc(timestamp)
    return bool(stamp and -300 <= (now-stamp).total_seconds() <= 3*3600)


def contract(game, entity, market, line, side, decimal, other_decimal, probability,
             projection, push, book, quoted_at, features, sample, why, risk, now, split=None,
             base_probability=None):
    if decimal is None or decimal<=1: return None
    devig=(1/decimal)/(1/decimal+1/other_decimal) if other_decimal and other_decimal>1 else None
    conditional=probability/(1-push) if probability is not None and push<1 else None
    edge=conditional-devig if conditional is not None and devig is not None else None
    identifier='|'.join(map(str,['mlb',game['game_pk'],entity['id'],market,line,side,book]))
    kind='game_line' if market in ('moneyline','spread','total') else 'pitcher_prop' if market=='pk' else 'hr' if market=='hr' else 'player_prop'
    count=min([v for k,v in sample.items() if k.endswith('_bf') or k.endswith('_pa')]+[600])
    confidence='moderate' if count>=150 else 'low'
    if market in ('moneyline','spread','total') and min(sample.get('home_games',0),sample.get('away_games',0))<30: confidence='low'
    description=f"{entity['name']} · {side.title()}"+(f' {line:g}' if line is not None else '')+f' {market}'
    evidence_why=list(why[:3])
    if split and split.get('relative_change'):
        evidence_why.append(f"Minor applicable-split adjustment: {split['relative_change']:+.2%}; cap ±3%.")
    return {'schema_version':1,'id':identifier,'sport':'mlb','game_pk':game['game_pk'],
        'event':{'id':str(game['game_pk']),'start':game['time'],'home':game['home'],'away':game['away']},
        'entity':entity,'opportunity_type':kind,'market':market,'side':side,'bet_description':description,
        'market_line':line,'price':{'decimal':decimal,'american':decimal_to_american(decimal),'book':book,'quoted_at':quoted_at},
        'going_projection':projection,'going_probability':probability,'push_probability':push,
        'conditional_win_probability':conditional,'devigged_market_probability':devig,'devigged_edge':edge,
        'expected_return':probability*(decimal-1)-(1-probability-push) if probability is not None else None,
        'confidence_level':confidence,'sample_sizes':sample,'why':evidence_why,
        'main_risk':risk+(' No matching opposite-side quote: no devigged edge asserted.' if devig is None else ''),
        'parlay_fit':{'game_pk':game['game_pk'],'correlation_group':f"mlb:{game['game_pk']}",
                      'rule':'Prefer distinct game_pk; same-game legs are correlated. No independence claim or synthetic SGP price.'},
        'features':features,'split_overlap':split or {'relative_change':0,'source':None},
        'baseline_probability':base_probability,'model_version':VERSION,'observed_at':now.isoformat(),
        'freshness':'FRESH','qualification':'research',
        'validation_status':('informational only; original K model retained, single short-start adjustment rejected; historical Brier improvement is not forward betting validation'
                             if kind=='pitcher_prop' else 'provisional; retrospective diagnostics are not demonstrated betting profitability'),
        'source':{'board':'board.json','markets':'opportunity_markets.json','stats':'https://statsapi.mlb.com/api/v1'}}


def build(board, markets, schedule, scores, pitching, batting, calibration, now=None):
    now=now or datetime.now(timezone.utc)
    cutoff=board['slate_date']
    games=[]
    for g in schedule:
        start=utc(g.get('gameDate'))
        if g['officialDate']!=cutoff or not start or start<=now or g['status'].get('abstractGameState')!='Preview': continue
        if any(not g['teams'][s].get('probablePitcher',{}).get('id') for s in ('home','away')): continue
        games.append({'game_pk':g['gamePk'],'time':g['gameDate'],**{s:g['teams'][s]['team']['abbreviation'] for s in ('home','away')},
                      **{s+'_name':g['teams'][s]['team']['name'] for s in ('home','away')},
                      **{s+'_sp':g['teams'][s]['probablePitcher']['id'] for s in ('home','away')},
                      **{s+'_sp_name':g['teams'][s]['probablePitcher'].get('fullName','Starter') for s in ('home','away')}})
    # The precomputed split boards are consumed exactly once, never fetched/recomputed.
    split=board.get('split_overlaps') or {}
    valid_split=split.get('status')=='FRESH' and split.get('slate_date')==cutoff and str(split.get('window_end_exclusive',''))<=cutoff
    split_boards=split.get('boards',{}) if valid_split else {}
    split_map={(family,r['game_pk'],r['id']):r for family,rows in split_boards.items() for r in rows}
    players={(p['game_pk'],_norm_name(p['name'])):p for p in board.get('players',[])
             if p.get('lineup_status') not in ('out','bench','inactive') and p.get('lineup_spot')}
    pitchers={(p['game_pk'],_norm_name(p['name'])):p for p in board.get('pitcher_props',[])}
    game_context={g['game_pk']:g for g in board.get('game_projections',[])}
    pens={p['team']:p for p in board.get('bullpen_rankings',[])}
    models={}
    for g in games:
        sp={s:pitching_features(pitching.get(g[s+'_sp'],[]),cutoff) for s in ('home','away')}
        def pen(s):
            p=pens.get(g[s],{})
            era,ip=p.get('bp_era'),p.get('total_ip')
            # Defensive: a malformed value from the bullpen_rankings source
            # (wrong type, e.g. a string) must degrade this one team's input
            # to None rather than raising and killing the entire run.
            if not isinstance(era,(int,float)) or not isinstance(ip,(int,float)) or not ip:
                return None
            return {'runs':era*ip/9,'ip':ip,'measure':'earned runs (ERA)'}
        ctx=game_context.get(g['game_pk'],{})
        park=(ctx.get('home_breakdown') or {}).get('park_mult',1)
        model=game_projection(scores,g['home'],g['away'],cutoff,sp['home'],sp['away'],park,pen('home'),pen('away'))
        model['features']['park_source']=ctx.get('park_run_src','neutral_missing')
        model['features']['weather']=next((w for w in board.get('wx',[]) if w['game_pk']==g['game_pk']),None)
        models[g['game_pk']]=model
    offers=[]; rejected=Counter()
    if markets.get('games',{}).get('status')=='FRESH':
        for event in markets['games'].get('rows',[]):
            g=match_game(event,games)
            if not g: rejected['game_identity_or_started']+=1; continue
            model=models[g['game_pk']]
            for book in event.get('bookmakers',[]):
                for market in book.get('markets',[]):
                    key={'h2h':'moneyline','spreads':'spread','totals':'total'}.get(market['key'])
                    if not key: continue
                    quoted=market.get('last_update') or book.get('last_update')
                    if not quote_fresh(quoted,now): rejected['stale_game_quote']+=1; continue
                    for outcome in market.get('outcomes',[]):
                        line=outcome.get('point') if key!='moneyline' else None
                        if key!='moneyline' and not isinstance(line,(int,float)): continue
                        if key=='total':
                            side=outcome['name'].lower()
                            if side not in ('over','under'): continue
                            pair=[o for o in market['outcomes'] if o.get('point')==line and o['name'].lower()!=side]
                            prob=settlement_prob(model['total_dist'],line,side); projection=model['total']
                        else:
                            side='home' if outcome['name']==g['home_name'] else 'away' if outcome['name']==g['away_name'] else None
                            if not side: continue
                            pair=[o for o in market['outcomes'] if o['name']!=outcome['name'] and (key=='moneyline' or o.get('point')==-line)]
                            dist=model['margin_dist'] if side=='home' else {-n:p for n,p in model['margin_dist'].items()}
                            prob=settlement_prob(dist,-line if line is not None else 0); projection=(model['home_runs']-model['away_runs'])*(1 if side=='home' else -1)
                        other=pair[0].get('price') if len(pair)==1 else None
                        c=contract(g,{'id':'game','name':g['away']+' @ '+g['home']},key,line,side,outcome.get('price'),other,
                            prob['win'],projection,prob['push'],book['key'],quoted,model['features'],model['sample_sizes'],
                            [f"Run-rate projection: {g['away']} {model['away_runs']:.2f}, {g['home']} {model['home_runs']:.2f}.",
                             f"Scheduled starters: {g['away_sp_name']} vs {g['home_sp_name']}; recent workload and runs allowed enter the estimate.",
                             f"Published park/weather factor {model['features']['park_weather_multiplier']:.2f}; no split-overlap inputs."],
                            'Run counts are overdispersed; bullpen availability and the one-run extra-innings approximation add uncertainty.',now)
                        if c: offers.append(c)
    if markets.get('props',{}).get('status')=='FRESH':
        for row in markets['props'].get('rows',[]):
            market=MARKETS.get(row.get('market_key'))
            # Existing HR milestone 1+ is exactly over 0.5, never over 1.0.
            milestone=row.get('market_key')=='player_home_runs_alt' and row.get('line')==1 and '1 or more' in str(row.get('market','')).lower()
            if milestone: market='hr'
            if not market or row.get('period')!='FULL' or row.get('is_dfs_flat_payout') or row.get('dfs_normalized'): continue
            g=match_game(row,games)
            if not g: rejected['prop_identity_or_started']+=1; continue
            if not quote_fresh(row.get('last_update'),now): rejected['stale_prop_quote']+=1; continue
            p=(pitchers if market=='pk' else players).get((g['game_pk'],_norm_name(row.get('player','').split('(')[0])))
            if not p: rejected['player_not_on_slate']+=1; continue
            line=.5 if milestone else row.get('line')
            if not isinstance(line,(int,float)): continue
            if market=='pk':
                if p['id'] not in (g['home_sp'],g['away_sp']): rejected['not_actual_starter']+=1; continue
                opponents=[x for (game_id,_),x in players.items() if game_id==g['game_pk'] and x['team']!=p['team']]
                sr=[split_map[('HIT_OVERLAP',g['game_pk'],x['id'])] for x in opponents if ('HIT_OVERLAP',g['game_pk'],x['id']) in split_map]
                f=pitching_features(pitching.get(p['id'],[]),cutoff)
                model=pitcher_projection(f,lineup_rates([x['id'] for x in opponents],batting,cutoff),sr)
                why=[f"Season K rate {f['season_k']:.1%}; recent-start shrunk K rate {f['recent_k']:.1%}.",
                     f"Expected workload {f['expected_ip']:.1f} IP / {f['expected_bf']:.1f} batters; {f['recent_starts']} recent starts.",
                     f"Opposing lineup shrunk K rate {model['features']['opposing_lineup_k']:.1%} ({len(opponents)} hitters)."]
                risk='Early removal or lineup changes reduce opportunities; this K extension did not outperform the simpler baseline in the initial holdout.'
                sample=model['sample_sizes']; features=model['features']
            else:
                family={'hr':'HR_OVERLAP','hits':'HIT_OVERLAP','hrr':'HRR_OVERLAP'}[market]
                model=batter_projection(p,market,line,split_map.get((family,g['game_pk'],p['id'])),calibration)
                if not model: rejected['missing_existing_projection']+=1; continue
                sample={'batter_pa':sum(r['pa'] for r in batting.get(p['id'],[]) if r['date']<cutoff)}
                features={k:p.get(k) for k in ('heat','hit_gated','hrr_proj','sample','lineup_spot','lineup_status')}
                why=([f"Existing HR heat-to-observed-rate calibration; heat {p.get('heat')}.",str(p.get('why') or 'Published HR scoring is reused unchanged.')]
                     if market=='hr' else [(f"Contact-gated chance of at least one hit {p['hit_gated']['p_hit']:.1%}; expected PA {p['hit_gated']['xpa']:.2f}." if market=='hits'
                     else f"Existing expected hits {p['hrr_proj'].get('exp_hits',0):.2f}, runs {p['hrr_proj'].get('exp_runs',0):.2f}, RBIs {p['hrr_proj'].get('exp_rbi',0):.2f}; calibrated HRR mean {p['hrr_proj']['hrr']*.817:.2f}."),
                     f"Lineup spot {p['lineup_spot']}; {p['lineup_status']} lineup; season sample {sample['batter_pa']}."])
                risk='Lineup/playing time may change; existing calibration is not independently validated betting profitability.'
            for side in ('over','under'):
                if milestone and side=='under': continue
                price=_prop_price_to_decimal(row.get(side+'_price'))
                opposite=_prop_price_to_decimal(row.get(('under' if side=='over' else 'over')+'_price')) if not milestone else None
                if not price: continue
                prob=settlement_prob(model['distribution'],line,side)
                c=contract(g,{'id':str(p['id']),'name':p['name']},market,line,side,price,opposite,prob['win'],
                           model['projection'],prob['push'],row['bookmaker'],row['last_update'],features,sample,why,risk,now,
                           model['split_overlap'],settlement_prob(model['baseline_distribution'],line,side)['win'])
                if c: offers.append(c)
    unique={c['id']:c for c in sorted(offers,key=lambda c:c['price']['quoted_at'])}
    rows=sorted(unique.values(),key=lambda c:-(c['devigged_edge'] if c['devigged_edge'] is not None else -10))
    for rank,row in enumerate(rows,1): row['rank']=rank
    return {'schema_version':1,'model_version':VERSION,'generated_at':now.isoformat(),'slate_date':cutoff,
            'status':'FRESH' if all(markets.get(k,{}).get('status')=='FRESH' for k in ('props','games')) else 'FALLBACK',
            'opportunities':rows,'counts':dict(Counter(r['opportunity_type'] for r in rows)),
            'rejected':dict(rejected),'sources':{'board_generated_at':board.get('generated_at'),
            'props':{k:v for k,v in markets.get('props',{}).items() if k!='rows'},
            'games':{k:v for k,v in markets.get('games',{}).items() if k!='rows'}},
            'deferred_markets':['total_bases','runs','rbi','stolen_bases','pitcher outs/hits allowed'],
            'note':'All quoted eligible sides, not all asserted bets. Devig needs same-book opposite side; integer pushes are separate.'}


def retain_failed_sources(payload, previous, markets):
    """Repeatable failed-run merge: original quote/observation times stay intact."""
    fresh_ids={r['id'] for r in payload['opportunities']}
    for row in previous.get('opportunities',[]):
        category='games' if row['opportunity_type']=='game_line' else 'props'
        source=markets.get(category,{})
        if source.get('status')!='FRESH' and row['id'] not in fresh_ids:
            stamp=utc(row.get('price',{}).get('quoted_at'))
            now=utc(payload.get('generated_at'))
            payload['opportunities'].append(dict(row,freshness='STALE',
                stale_reason=source.get('error','source_not_fresh'),
                quote_age_seconds=max(0,(now-stamp).total_seconds()) if stamp and now else None))
    if not fresh_ids:
        payload['status']='STALE' if payload['opportunities'] else 'FAILED'
    payload['retained_stale_count']=sum(r['freshness']=='STALE' for r in payload['opportunities'])
    return payload


def freeze(payload, root):
    from etl.opportunity_models import valid_frozen_price
    root=Path(root)
    path=root/'opportunity_predictions.json'
    ledger=json.loads(path.read_text()) if path.exists() else {'schema_version':1,'records':[]}
    records={r['id']:r for r in ledger['records']}
    for row in payload['opportunities']:
        if not valid_frozen_price(row): continue
        if row['id'] in records: continue  # first pregame observation, never overwrite
        records[row['id']]={'schema_version':1,'id':row['id'],'sport':'mlb','event':row['event'],'entity':row['entity'],
            'signal_family':'MLB_'+row['opportunity_type'].upper(),'model_version':VERSION,'observed_at':row['observed_at'],
            'features':{'prediction':row,'split_overlap':row['split_overlap']},'sample_sizes':row['sample_sizes'],
            'rank':row['rank'],'strength':row['devigged_edge'],'source':row['source'],'outcome':None}
    ledger.update(generated_at=payload['generated_at'],records=list(records.values()))
    save(path,ledger)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',default='docs'); args=parser.parse_args()
    root=Path(args.root); now=datetime.now(timezone.utc)
    try:
        board=json.loads((root/'board.json').read_text(encoding='utf-8'))
        if (not utc(board.get('generated_at')) or not 0 <= (now-utc(board['generated_at'])).total_seconds() <= 30*3600
                or board['slate_date']!=now.astimezone(ZoneInfo('America/New_York')).date().isoformat()):
            raise ValueError('Board stale')
        markets=json.loads((root/'opportunity_markets.json').read_text(encoding='utf-8'))
        for key in ('props','games'):
            if not quote_fresh(markets.get(key,{}).get('updated_at'),now): markets.setdefault(key,{})['status']='STALE'
        api=MLB(); date=board['slate_date']; season=date[:4]
        schedule=api.schedule(season+'-03-01',date)
        ids=[g['teams'][s].get('probablePitcher',{}).get('id') for g in schedule if g['officialDate']==date for s in ('home','away')]
        pitching=api.logs(ids,season)
        batting=api.logs([p['id'] for p in board['players']],season,'hitting')
        calibration=json.loads((root/'backtest.json').read_text()).get('calib',{})
        payload=build(board,markets,schedule,final_scores(schedule),pitching,batting,calibration,now)
        if payload['status']!='FRESH' and (root/'opportunities.json').exists():
            previous=json.loads((root/'opportunities.json').read_text())
            retain_failed_sources(payload,previous,markets)
        payload['source_calls']=api.calls
        save(root/'opportunities.json',payload); freeze(payload,root)
        from etl.signal_records import build as ledger
        ledger(root)
        from etl.grade_opportunities import publish_ledger
        publish_ledger(root)
        print(json.dumps({k:payload[k] for k in ('status','counts','rejected','source_calls')}))
        if payload['status'] in ('FAILED','STALE','FALLBACK'):
            raise SystemExit('One or more required market sources unavailable; source-specific freshness retained')
    except Exception as error:
        import traceback
        path=root/'opportunities.json'
        old=json.loads(path.read_text()) if path.exists() else {'opportunities':[]}
        # Capture the real message and the last few frames, not just the bare
        # exception class name -- a bare "TypeError" with no message or
        # location is undiagnosable from the deployed JSON alone.
        tb_lines=traceback.format_exception(type(error),error,error.__traceback__)
        old.update(status='STALE' if old['opportunities'] else 'FAILED',last_attempt=now.isoformat(),
                   error=type(error).__name__,error_message=str(error) or None,
                   error_traceback=''.join(tb_lines[-6:]))
        for row in old['opportunities']: row['freshness']='STALE'
        save(path,old)
        raise


if __name__=='__main__': main()
