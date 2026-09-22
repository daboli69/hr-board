"""Batched official MLB inputs; daily cache and immutable final boxscores.

No additional BallparkPal, Open-Meteo or Statcast requests. Numeric StatsAPI
limits are not published: requests are serialized at <= 1/second, timeout 25s,
and bounded to two attempts. All date cutoffs are applied again locally.
"""
import hashlib
import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone
import requests
from etl.opportunity_models import innings

BASE = 'https://statsapi.mlb.com/api/v1'


def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


class MLB:
    def __init__(self, cache='.cache/opportunities'):
        self.cache = Path(cache)
        self.session = requests.Session()
        self.last = 0
        self.calls = 0

    def get(self, path, params=None, immutable=False):
        params = params or {}
        day = 'final' if immutable else datetime.now(timezone.utc).date().isoformat()
        key = hashlib.sha256(json.dumps([path, params, day], sort_keys=True).encode()).hexdigest()
        file = self.cache / (key+'.json')
        if file.exists():
            return json.loads(file.read_text(encoding='utf-8'))
        for attempt in range(2):
            time.sleep(max(0, 1-(time.monotonic()-self.last)))
            self.last = time.monotonic()
            self.calls += 1
            response = self.session.get(BASE+path, params=params, timeout=25)
            if response.status_code in (429, 500, 502, 503, 504) and attempt == 0:
                time.sleep(5)
                continue
            response.raise_for_status()
            body = response.json()
            save(file, body)
            return body
        raise RuntimeError('MLB source unavailable')

    def schedule(self, start, end):
        body = self.get('/schedule', {'sportId':1, 'startDate':start, 'endDate':end,
                                    'hydrate':'probablePitcher,team,lineups'})
        return [g for d in body.get('dates',[]) for g in d['games'] if g.get('gameType')=='R']

    def logs(self, ids, season, group='pitching'):
        result = {}
        ids = sorted(set(int(i) for i in ids if i))
        for begin in range(0,len(ids),40):
            people = self.get('/people', {'personIds':','.join(map(str,ids[begin:begin+40])),
                      'hydrate':f'stats(group=[{group}],type=[gameLog],season={season})'})
            for person in people.get('people',[]):
                rows=[]
                for stats in person.get('stats',[]):
                    for split in stats.get('splits',[]):
                        s=split['stat']
                        if group=='pitching':
                            rows.append({'date':split['date'], 'game_pk':split['game']['gamePk'],
                                         'started':bool(s.get('gamesStarted')), 'bf':s.get('battersFaced',0),
                                         'k':s.get('strikeOuts',0), 'ip':innings(s.get('inningsPitched')),
                                         'runs':s.get('runs',0)})
                        else:
                            rows.append({'date':split['date'],'pa':s.get('plateAppearances',0),'k':s.get('strikeOuts',0)})
                result[person['id']]=rows
        return result

    def box(self, game_pk):
        # Caller must first verify official Final status before immutable caching.
        return self.get(f'/game/{game_pk}/boxscore', immutable=True)


def final_scores(schedule):
    return sorted([{'game_pk':g['gamePk'],'date':g['officialDate'],
             'home':g['teams']['home']['team']['abbreviation'], 'away':g['teams']['away']['team']['abbreviation'],
             'home_runs':g['teams']['home']['score'],'away_runs':g['teams']['away']['score']}
             for g in schedule if g.get('status',{}).get('abstractGameState')=='Final'
             and all('score' in g['teams'][s] for s in ('home','away'))], key=lambda g:(g['date'],g['game_pk']))


def lineup_rates(ids, logs, cutoff):
    return [{'id':i,'pa':sum(r['pa'] for r in logs.get(i,[]) if r['date']<cutoff),
             'k':sum(r['k'] for r in logs.get(i,[]) if r['date']<cutoff)} for i in ids]
