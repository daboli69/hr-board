"""Pure, cutoff-safe MLB opportunity models. Never imports or changes build_board.

Run-rate multiplicative matchup model with empirical-Bayes shrinkage; count
uncertainty is negative-binomial, not a normal approximation to discrete lines.
Constants are versioned, not retuned from the predictions being evaluated.
"""
import math
from datetime import datetime, timezone

VERSION = 'yard-opportunities-1'
SPLIT_CAP = .03  # maximum relative change; not three probability points


def utc(value):
    try:
        result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return result.astimezone(timezone.utc) if result.tzinfo else None
    except (TypeError, ValueError):
        return None


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def valid_frozen_price(prediction):
    """Freshness at capture, NOT price age at settlement time."""
    observed=utc(prediction.get('observed_at'))
    quoted=utc((prediction.get('price') or {}).get('quoted_at'))
    kickoff=utc((prediction.get('event') or {}).get('start'))
    return bool(prediction.get('freshness')=='FRESH' and observed and quoted and kickoff
                and observed<kickoff and quoted<kickoff
                and -300 <= (observed-quoted).total_seconds() <= 10800)


def shrunk(total, exposure, prior, prior_exposure):
    return (total + prior * prior_exposure) / (exposure + prior_exposure)


def innings(value):
    whole, _, outs = str(value or '0').partition('.')
    return int(whole) + int(outs or 0) / 3


def count_dist(mean, dispersion=1., maximum=100):
    """NB var/mean parameterization; Poisson at dispersion=1."""
    mean = max(0., float(mean))
    if mean == 0:
        return {0: 1.}
    if dispersion <= 1:
        values = [math.exp(-mean)]
        for k in range(1, maximum + 1):
            values.append(values[-1] * mean / k)
    else:
        r, q = mean / (dispersion - 1), 1 / dispersion
        values = [q ** r]
        for k in range(1, maximum + 1):
            values.append(values[-1] * (k - 1 + r) * (1 - q) / k)
    total = sum(values)
    if total < .999999:
        raise ValueError('Count distribution truncated')
    return {k: p / total for k, p in enumerate(values)}


def settlement_prob(dist, line, side='over'):
    win = sum(p for n, p in dist.items() if (n > line if side == 'over' else n < line))
    push = dist.get(line, 0.)
    return {'win': win, 'push': push, 'conditional_win': win / (1 - push) if push < 1 else None}


def split_effect(row):
    if not row:
        return {'relative_change': 0., 'cap': SPLIT_CAP, 'source': None}
    confidence = clip(float(row.get('sample_confidence') or 0) / 100, 0, 1)
    # Neutral strength is 100 (ratio × ratio × 100), NOT 50.
    change = SPLIT_CAP * confidence ** 2 * clip(float(row['overlap_strength']) / 100 - 1, -1, 1)
    return {'relative_change': change, 'cap': SPLIT_CAP, 'sample_confidence': confidence,
            'strength': row['overlap_strength'], 'source': row.get('signal_family'),
            'batter_stats': row.get('batter_stats'), 'pitcher_stats': row.get('pitcher_stats')}


def team_rates(games, team, cutoff):
    rows = []
    for g in games:
        if g['date'] >= cutoff or team not in (g['home'], g['away']):
            continue
        home = g['home'] == team
        rows.append((g['home_runs'] if home else g['away_runs'], g['away_runs'] if home else g['home_runs']))
    recent = rows[-14:]
    return {'games': len(rows), 'rs': shrunk(sum(r[0] for r in rows), len(rows), 4.4, 20),
            'ra': shrunk(sum(r[1] for r in rows), len(rows), 4.4, 20),
            'recent_rs': shrunk(sum(r[0] for r in recent), len(recent), 4.4, 20)}


def pitching_features(logs, cutoff):
    rows = [r for r in logs if r['date'] < cutoff and r.get('started')]
    rows.sort(key=lambda r: (r['date'], r.get('game_pk', 0)))
    recent = rows[-5:]
    bf = sum(r['bf'] for r in rows)
    k = sum(r['k'] for r in rows)
    ip = sum(r['ip'] for r in rows)
    rbf = sum(r['bf'] for r in recent)
    season_k = shrunk(k, bf, .225, 150)
    recent_k = shrunk(sum(r['k'] for r in recent), rbf, season_k, 100)
    # Overlapping recent data is a bounded recency tilt, not independent evidence.
    weight = min(.25, rbf / (rbf + 300))
    return {'bf': bf, 'ip': ip, 'starts': len(rows), 'recent_starts': len(recent),
            'season_k': season_k, 'recent_k': recent_k, 'recent_weight': weight,
            'k_rate': season_k * (1 - weight) + recent_k * weight,
            'ra9': 9 * shrunk(sum(r['runs'] for r in rows), ip, 4.4 / 9, 60),
            'recent_ra9': 9 * shrunk(sum(r['runs'] for r in recent), sum(r['ip'] for r in recent), 4.4 / 9, 35),
            'expected_ip': shrunk(sum(r['ip'] for r in recent), len(recent), 5., 2),
            'expected_bf': shrunk(sum(r['bf'] for r in recent), len(recent), 22., 2),
            'workloads': [r['bf'] for r in recent]}


def game_projection(games, home, away, cutoff, home_sp, away_sp, park=1., home_pen=None, away_pen=None):
    h, a = team_rates(games, home, cutoff), team_rates(games, away, cutoff)
    def runs(off, defense, starter, pen):
        offense = .85 * off['rs'] + .15 * off['recent_rs']
        ip = clip(starter['expected_ip'], 1, 7.5)
        sp = .85 * starter['ra9'] + .15 * starter['recent_ra9']
        pen_rate = shrunk(pen['runs'], pen['ip'], 4.4 / 9, 60) * 9 if pen and pen.get('ip') else defense['ra']
        allowed = .75 * (sp * ip + pen_rate * (9 - ip)) / 9 + .25 * defense['ra']
        return clip(offense * allowed / 4.4 * park, 1.2, 12.)
    hm, am = runs(h, a, away_sp, away_pen), runs(a, h, home_sp, home_pen)
    # Venue advantage is a restrained run-rate multiplier; park already includes weather.
    hm, am = hm * 1.02, am * .98
    hd, ad = count_dist(hm, 1.5), count_dist(am, 1.5)
    margin, total = {}, {}
    for hn, hp in hd.items():
        for an, ap in ad.items():
            margin[hn-an] = margin.get(hn-an, 0) + hp*ap
            total[hn+an] = total.get(hn+an, 0) + hp*ap
    # Nine-inning ties need a winner. Approximate extras by run-strength odds;
    # move tie mass to ±1 for a coherent full-game margin and total distribution.
    p_extra = hm / (hm + am)
    tie = margin.pop(0, 0)
    margin[1] += tie * p_extra
    margin[-1] += tie * (1-p_extra)
    for n in hd.keys() & ad.keys():
        mass = hd[n] * ad[n]
        total[2*n] -= mass
        total[2*n+1] = total.get(2*n+1, 0) + mass
    return {'home_runs': hm, 'away_runs': am, 'total': sum(n*p for n,p in total.items()),
            'home_probability': sum(p for n,p in margin.items() if n>0),
            'margin_dist': margin, 'total_dist': total,
            'features': {'home_rates': h, 'away_rates': a, 'home_starter': home_sp,
                         'away_starter': away_sp, 'home_bullpen': home_pen, 'away_bullpen': away_pen,
                         'park_weather_multiplier': park},
            'sample_sizes': {'home_games': h['games'], 'away_games': a['games'],
                             'home_starter_bf': home_sp['bf'], 'away_starter_bf': away_sp['bf']}}


def pitcher_projection(features, lineup, split_rows=(), short_start_smoothing=False):
    # Opponent lineup is only actual/projected batters, not every player on a roster.
    pa, ks = sum(r['pa'] for r in lineup), sum(r['k'] for r in lineup)
    opposing_k = shrunk(ks, pa, .225, 250)
    p = clip(features['k_rate'] * opposing_k / .225, .06, .5)
    # The HIT overlap K components are relevant; its hit/AVG strength is not.
    effects = []
    for r in split_rows:
        bk, pk = r['batter_stats'].get('k_pct'), r['pitcher_stats'].get('k_pct')
        if bk is not None and pk is not None:
            c = clip(r.get('sample_confidence', 0)/100, 0, 1)
            effects.append(SPLIT_CAP*c*c*clip((bk/22.5)*(pk/22.5)-1, -1, 1))
    delta = sum(effects)/len(effects) if effects else 0.
    baseline_mean = features['expected_bf'] * p
    mean = baseline_mean * (1+delta)
    # Poisson mixed over observed last-five workloads admits early exits and long starts.
    workloads = features['workloads'] or [features['expected_bf']]
    if short_start_smoothing and features['expected_ip'] < 4.5:
        # Single follow-up candidate: remove the second workload-variance layer
        # for pregame short-start profiles. Mean and every K-rate input unchanged.
        workloads = [features['expected_bf']]
    average = sum(workloads)/len(workloads)
    dist, baseline_dist = {}, {}
    for bf in workloads:
        for k, probability in count_dist(mean * (bf+22)/(average+22), 1.15).items():
            dist[k] = dist.get(k, 0) + probability/len(workloads)
        for k, probability in count_dist(baseline_mean * (bf+22)/(average+22), 1.15).items():
            baseline_dist[k] = baseline_dist.get(k, 0) + probability/len(workloads)
    return {'projection': mean, 'distribution': dist, 'baseline_distribution': baseline_dist, 'baseline_projection': baseline_mean,
            'features': dict(features, opposing_lineup_k=opposing_k, lineup_players=len(lineup)),
            'sample_sizes': {'pitcher_bf': features['bf'], 'lineup_pa': pa, 'recent_starts': features['recent_starts']},
            'split_overlap': {'relative_change': delta, 'cap': SPLIT_CAP, 'source': 'SPLIT_HIT_OVERLAP K components' if effects else None,
                              'rows': list(split_rows)}}


def hr_probability(heat, calibration):
    """Exact calibratedHRprob interpolation already used by Yard, no score rebuild."""
    if heat is None:
        return None
    b = max(0, min(90, math.floor(heat/10)*10))
    def rate(e):
        return e['hr']/e['n'] if e and e.get('n',0)>=25 else None
    lo, hi = rate(calibration.get(str(b))), rate(calibration.get(str(min(90,b+10))))
    if lo is None: return hi
    if hi is None: return lo
    return lo+(heat-b)/10*(hi-lo)


def batter_projection(player, market, line, split=None, calibration=None):
    effect = split_effect(split)
    factor = 1 + effect['relative_change']
    if market == 'hr':
        base = hr_probability(player.get('heat'), calibration or {})
        if base is None or line != .5: return None
        return {'projection': None, 'probability': clip(base*factor,0,1), 'baseline_probability': base,
                'distribution': {0:1-clip(base*factor,0,1),1:clip(base*factor,0,1)},
                'baseline_distribution': {0:1-base,1:base}, 'split_overlap': effect}
    if market == 'hits':
        hit = player.get('hit_gated') or {}
        if hit.get('p_hit') is None or not hit.get('xpa'): return None
        # Recover per-PA probability from the EXISTING fractional-PA hit probability.
        n, frac = int(hit['xpa']), hit['xpa'] % 1
        lo, hi = 0., 1.
        for _ in range(45):
            p=(lo+hi)/2
            if 1-(1-p)**n*(1-frac*p)<hit['p_hit']: lo=p
            else: hi=p
        p0=(lo+hi)/2
        p=clip(p0*factor,0,1)
        dist={k:(1-frac)*math.comb(n,k)*p**k*(1-p)**(n-k) if k<=n else 0 for k in range(n+2)}
        for k in dist: dist[k]+=frac*math.comb(n+1,k)*p**k*(1-p)**(n+1-k)
        base_dist={k:(1-frac)*(math.comb(n,k)*p0**k*(1-p0)**(n-k) if k<=n else 0)+frac*math.comb(n+1,k)*p0**k*(1-p0)**(n+1-k) for k in range(n+2)}
        mean=hit['xpa']*p
    elif market == 'hrr':
        hrr=player.get('hrr_proj') or {}
        if hrr.get('hrr') is None: return None
        # Preserve existing 0.817 mean correction and 1.8 variance/mean calibration.
        mean=hrr['hrr']*.817*factor
        dist=count_dist(mean,1.8)
        base_dist=count_dist(hrr['hrr']*.817,1.8)
    else: return None
    return {'projection': mean, 'probability': settlement_prob(dist,line)['win'],
            'baseline_probability': settlement_prob(base_dist,line)['win'],
            'distribution': dist, 'baseline_distribution': base_dist, 'split_overlap': effect}
