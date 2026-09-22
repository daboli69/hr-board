# MLB opportunities v1

## Scope and provenance

`etl/build_opportunities.py` is independent of `etl/build_board.py`. HR scoring,
hit/HRR scoring and pure split boards are unchanged. All quoted eligible sides
are exported; a model disagreement is explicitly research, not proof of profit.
Identity requires both teams, reported game time, the slate player and actual
probable starter. Wrong-team feed rows, unresolved doubleheaders, inactive
players, non-full-game and synthetic DFS prices are rejected and counted.

## Decisions

- **Hits:** reuse `hit_gated.p_hit` and fractional expected PA exactly at 0.5.
  Recover the existing implied per-PA Bernoulli parameter to price other hit
  thresholds with the same fractional-binomial distribution. This is a
  distribution extension, not a new contact model.
- **HRR:** reuse the existing mean, its 0.817 correction and NB variance/mean
  1.8. Its historical fitted correction is NOT claimed independent validation.
- **HR:** reuse Yard's existing `calibratedHRprob` interpolation over its heat
  calibration buckets. Do not replace heat, add another power model or invent
  a fallback probability when the calibration bucket is missing.
- **Split cap:** +/-3% relative rate/probability, multiplied by squared sample
  confidence. Neutral strength is 100, not 50. This is conservative
  regularization of correlated secondary evidence, not an empirically established
  lift. Hit/HRR/HR use the corresponding already-built board. Ks use only its
  applicable batter/pitcher K rates. No split input enters game lines. Baseline
  and adjusted probabilities are both frozen for prospective ablation.
- **Game lines:** shrink team runs scored/allowed toward 4.4 with 20 prior games;
  multiply offensive run strength by opponent run-allowance strength. The latter
  combines scheduled starter RA9/workload, bullpen ERA and team RA. Recent
  scoring and starter RA9 supply a bounded 15% recency tilt. Existing run-park
  multipliers (including weather where supplied) are reused, without another
  weather fetch or a second weather boost. Independent overdispersed count
  distributions produce ML, every quoted run line and total. A tied nine-inning
  score is approximated by one extra run allocated by relative run strength.
  This approximation, and missing park/bullpen observations, remain visible.
- **Pitcher Ks:** season K/BF shrunk by 150 prior BF; recent five starts shrunk
  toward that rate and bounded at 25% recency weight. Real opposing lineup K/PA
  is shrunk with 250 prior PA. Last-five BF/IP supplies expected workload;
  workload mixtures admit short starts. Counts are overdispersed (variance/mean
  1.15). The initial K extension did not beat its simpler season-rate baseline;
  it remains explicitly provisional, not a validated edge.

The framework uses standard empirical-Bayes regression to the mean and discrete
count models. Prior exposures and dispersion constants are fixed, versioned
regularizers, not fitted to this holdout. Run-count overdispersion is supported
by [Albert, Beyond runs expectancy](https://journals.sagepub.com/doi/10.3233/JSA-140001).
The comparison baseline uses Pythagorean run strength and
[log5](https://sabr.org/journal/article/probabilities-of-victory-in-head-to-head-team-matchups/).
No assertion is made that these fixed constants are optimal.

## Output and settlement

`opportunities.json` contains every required contract field, actual book/quote
timestamp, features, samples, model version, split contribution, same-game
correlation identity, and rejection counts. Devigging uses BOTH sides of the
SAME book/line, never cross-book best prices. Missing opposite quotes yield
null devig/edge, not a manufactured probability. Integer-line pushes are
separate; the market comparison uses probability conditional on no push.

`opportunity_predictions.json` freezes the first pregame observation per
game/player/market/line/side/book. `grade_opportunities.py` only settles official
Final games and verified appearances. Missing stats remain pending. Research
settlement assumes a starter requirement for Ks and an appearance requirement
for batters; book-specific shortened-game/starting-batter rules may differ.
All four families merge into the existing sport-agnostic `signal_tracker.json`.
No result changes the frozen prediction or retunes model weights.

## Measured validation

`opportunity_backtest.json` is reproducible with
`python -m etl.backtest_opportunities --games 60`.

| Diagnostic | N | Brier | Baseline Brier |
|---|---:|---:|---:|
| Home moneyline | 60 | 0.212925 | 0.223183 |
| Total over 8.5 | 60 | 0.252199 | — |
| Home -1.5 | 60 | 0.247518 | — |
| Pitcher K over 4.5 | 109 | 0.247371 | 0.241593 |
| HR, chronological existing interpolator | 1,163 | 0.093432 | — |

Total MAE 2.7225 runs; K MAE 1.9858. The report also includes log loss, ECE,
calibration bins and approximate uncertainty intervals. The small sample does
not establish improvement or profitability. Recent-score tests use official
results and exclude ALL performance on/after each test game date. Retrospective
starter/lineup identities are conditional actual identities, not reconstructed
announcement-time information. Historical park/weather and bullpen availability
were not frozen: those live-only adjustments are not covered by this backtest.
Threshold diagnostics are NOT historical sportsbook-price or ROI backtests.

Existing immutable player outcome tracking supplies 1,170 chronological holdout
player-games: hits >=1 observed 55.56%; HRR >=2 observed 41.62%. Old snapshots
did not retain contact-gated/HRR probabilities, so genuine historical calibration
for those probabilities cannot be reconstructed without inventing data. New
records retain them. Historical split ablation is likewise unavailable; its
prospective baseline/adjusted forecasts are recorded now.

## Verified quotas and fetch budget (2026-09-22)

Authenticated `/v1/usage`: **Pro, 100,000 credits/month**, 3,781 used and 96,219
remaining before this task's probes. Paid response headers report no ordinary
per-second cap; account metadata reports a 10,000/sec safety ceiling. This job
is sequential and nowhere near it. No credentials or account email are stored.

Current old code: three requests/run (props, ML+total, spread), nominal six
credits/run. The unfiltered props call silently stopped at 5,000 rows.
New: one shared full-game props query for existing DK/Fanatics/FanDuel coverage
and one bundled game-market query. The verified props query returned 1,979 rows
without truncation; both live requests billed **3 credits**. Expanding prop
books produced an explicit provider truncation flag, which is rejected rather
than silently accepted. All required market families are preserved.

Normal budget: **2 requests / 6 credits per run; 16 requests / 48 credits per
day; 480 requests / 1,440 credits per 30-day month (1.44% of Pro)**. Pagination
is bounded at three pages; transport retries at three attempts. Extreme
all-pages/all-retries bound is 12 requests / 36 credits per run, 8,640 credits
per 30 days. Actual credit/remaining headers are persisted. Account usage also
includes other applications; this is the MLB job's budget, not a forecast of
unknown consumers. No run-frequency reduction was necessary.

Each scheduled run is a real refresh: the documented last-update metadata
arrives in the paid response, so no imaginary free unchanged-price check is
used. Models share those responses and never fetch odds independently.
Official MLB season schedules/logs are batched and daily cached, <=1 request/sec;
final boxscores are immutable-cached. Initial live feature build used 13 StatsAPI
calls; same-day repeat used zero. The 60-game one-time backtest used 73 calls.
No published contractual numeric StatsAPI/Statcast limit was established; they
are not called “unlimited.” New BallparkPal/Open-Meteo/Statcast calls: **zero**.
BallparkPal account quota was not available to verify. Open-Meteo's free tier
allows <10,000/day, 5,000/hour, 600/minute and is **non-commercial only**;
[subscriptions/ads require appropriate commercial terms](https://open-meteo.com/en/terms).
No paid service was added or purchased.

Both GitHub repositories were verified public. Standard hosted runners have
[free public-repository minutes](https://docs.github.com/en/billing/concepts/product-billing/github-actions),
subject to normal concurrency/runtime limits. No local service is introduced.

## Schedules and device independence

- **Update HR Board:** `30 10 * * *` plus `17 0,2,14,16,18,20,22 * * *` UTC;
  eight in-season runs/day. EDT: 06:30, 10:17, 12:17, 14:17, 16:17, 18:17,
  20:17, 22:17. Existing HR schedule unchanged. Offseason weekly season gate.
- **Grade HR Board:** `0 11,17,23 * * *` UTC (07:00/13:00/19:00 EDT), plus
  Monday `40 10 * * 1` offseason retry. Official MLB opportunity grading shares
  this existing workflow and catches up pending final games.
- **Refresh Going Long data:** `20 10,22 * * *` UTC (06:20/18:20 EDT).
  Existing NFL/NCAA results, frozen tracker and multisport publication remain
  here. Independent tracker/publication steps now still run after unrelated
  source failures; genuine failures remain visible in Actions.

These are UTC cron schedules (EST display shifts an hour); GitHub may delay or
skip individual scheduled starts. Manual workflow dispatch is also available.
Observed hosted MLB grading succeeded; the latest football run's settlement,
tracker and static publisher succeeded although its final source report failed.
All grading and publication run in GitHub Actions with repository secrets, not
Travis's machine, an open browser or a local database.

## Final live provider check

The initial live bundle returned 15 game events and all four opportunity
families were built successfully. A later production-entry-point check refreshed
props but the game endpoint returned **HTTP 502, HTML, Retry-After: 60**, with
no billed-credit header. This is recorded as a current external availability
limitation, not a successful fresh game-odds refresh. Gateway retries are bounded
and honor Retry-After up to 60 seconds; longer waits fail/degrade rather than
retry early. Game offers retain original timestamps and STALE status while
independent fresh props remain usable. The next healthy scheduled response
restores game markets automatically; no user device is involved.
