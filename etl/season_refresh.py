"""Gate costly scheduled builds using MLB's published schedule. Manual runs always work."""
import datetime as dt
import json
import os
import urllib.request

def should_refresh(total_games, mode, now):
    return total_games > 0 or (mode == 'weekly' and now.weekday() == 0 and now.hour == 10)

def main():
    now=dt.datetime.now(dt.timezone.utc)
    start=(now-dt.timedelta(days=3)).date();end=(now+dt.timedelta(days=7)).date()
    url=f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start}&endDate={end}'
    try:
        with urllib.request.urlopen(url,timeout=20) as response:data=json.load(response)
        count=data['totalGames']
        if not isinstance(count,int) or count<0:raise ValueError('Invalid schedule count')
        run=should_refresh(count,os.environ.get('REFRESH_MODE','weekly'),now)
        print(f'MLB schedule: {count} games from {start} through {end}; refresh={run}')
    except Exception as exc:
        # An outage must not accidentally shut off the in-season board.
        run=True
        print(f'Schedule unavailable ({type(exc).__name__}); retaining scheduled refresh')
    with open(os.environ['GITHUB_OUTPUT'],'a',encoding='utf-8') as output:output.write(f'run={str(run).lower()}\n')
if __name__=='__main__':main()
