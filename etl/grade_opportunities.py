"""Scheduled official-final settlements; independent of users and browsers."""
import json
from collections import defaultdict
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from etl.opportunity_data import MLB, save
from etl.opportunity_models import utc, valid_frozen_price


def settle(prediction, game, box):
    if game.get('status',{}).get('abstractGameState')!='Final': return None
    if not valid_frozen_price(prediction): return None
    if not utc(prediction.get('observed_at')) or utc(prediction['observed_at'])>=utc(prediction['event']['start']):
        raise ValueError('Prediction is not pregame')
    market=prediction['market']; side=prediction['side']; line=prediction['market_line']
    if market in ('moneyline','spread','total'):
        h=game['teams']['home'].get('score'); a=game['teams']['away'].get('score')
        if h is None or a is None: return None
        actual=h+a if market=='total' else h-a if side=='home' else a-h
        difference=actual-line if market=='total' else actual+(line or 0)
        if market=='total' and side=='under': difference=-difference
    else:
        player=next((p for team in box.get('teams',{}).values() for p in team.get('players',{}).values()
                     if str(p.get('person',{}).get('id'))==str(prediction['entity']['id'])),None)
        if not player: return None
        stats=player.get('stats',{}).get('pitching' if market=='pk' else 'batting',{})
        played=stats.get('gamesStarted') if market=='pk' else stats.get('plateAppearances')
        if not played:
            # Do not invent DNP from absent data; verified bench or nonstarting
            # pitcher is explicitly voided under this research settlement policy.
            verified=(market=='pk' and 'gamesStarted' in stats) or player.get('gameStatus',{}).get('isOnBench') is True
            return {'status':'void','actual':None,'reason':'No qualifying appearance'} if verified else None
        keys={'pk':['strikeOuts'],'hr':['homeRuns'],'hits':['hits'],'hrr':['hits','runs','rbi']}[market]
        if any(k not in stats for k in keys): return None
        actual=sum(stats[k] for k in keys)
        difference=(actual-line)*(1 if side=='over' else -1)
    status='win' if difference>0 else 'loss' if difference<0 else 'push'
    return {'status':status,'actual':actual,'unit_return':prediction['price']['decimal']-1 if status=='win' else -1 if status=='loss' else 0}


def publish_ledger(root):
    root=Path(root); path=root/'signal_tracker.json'
    ledger=json.loads(path.read_text()) if path.exists() else {'schema_version':1,'records':[]}
    records={r['id']:r for r in ledger['records']}
    source=root/'opportunity_predictions.json'
    if source.exists(): records.update({r['id']:r for r in json.loads(source.read_text())['records']})
    ledger.update(generated_at=datetime.now(timezone.utc).isoformat(),records=list(records.values()))
    save(path,ledger)


def run(root='docs'):
    root=Path(root); path=root/'opportunity_predictions.json'
    if not path.exists(): return
    ledger=json.loads(path.read_text()); pending=defaultdict(list)
    for r in ledger['records']:
        if not valid_frozen_price(r.get('features',{}).get('prediction',{})):
            r['grading_status']='ineligible_stale_or_unverifiable_capture'
            continue
        if r.get('outcome') is None: pending[r['event']['id']].append(r)
    api=MLB(); errors=[]; graded=0
    # One schedule range and one final boxscore per game, reused across every offer.
    dates=[r['event']['start'][:10] for rows in pending.values() for r in rows]
    # West-coast games can start on the next UTC date while their official MLB
    # date remains yesterday. Include that date instead of leaving them pending.
    schedule=api.schedule((date.fromisoformat(min(dates))-timedelta(days=1)).isoformat(),max(dates)) if dates else []
    games={str(g['gamePk']):g for g in schedule}
    for gid,records in pending.items():
        g=games.get(gid,{})
        if g.get('status',{}).get('abstractGameState')!='Final': continue
        try:
            box=api.box(gid)
            for r in records:
                result=settle(r['features']['prediction'],g,box)
                if result:
                    r['outcome']={'values':result,'observed_at':datetime.now(timezone.utc).isoformat(),
                                  'source':f'https://statsapi.mlb.com/api/v1/game/{gid}/boxscore'}
                    graded+=1
        except Exception as e: errors.append({'game_pk':gid,'error':type(e).__name__})
    ledger.update(generated_at=datetime.now(timezone.utc).isoformat(),errors=errors)
    save(path,ledger); publish_ledger(root)
    from etl.opportunity_followup import update_roi
    update_roi(root)
    print(json.dumps({'graded_offers':graded,'pending_games':len(pending),'source_calls':api.calls,'errors':errors}))
    if errors: raise RuntimeError('Some official outcomes unavailable; prior settlements preserved')


if __name__=='__main__': run()
