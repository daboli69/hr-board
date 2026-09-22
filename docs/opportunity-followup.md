# Opportunities v1 follow-up

Implementation and reproducible results: `etl/opportunity_followup.py`,
`opportunity_followup.json`, `tests/test_opportunity_followup.py`.
Original `opportunity_backtest.json` is preserved unchanged.

## 1. Same 60 games: missing baseline comparisons

Lower Brier is better. Exactly the original game IDs and date-exclusive features
are reused, not whichever 60 games happen to be latest when the script runs.

| Market diagnostic | V1 model | Pythagorean/log5 baseline |
|---|---:|---:|
| Home ML | 0.212925 | 0.223183 |
| Home -1.5 | 0.247518 | 0.255418 |
| Total over 8.5 | 0.252199 | 0.246104 |

Log5 gives only winner probability. For a score-based baseline, the unadjusted
shrunk season RS/RA matchup rates supply the score scale and v1's fixed NB
dispersion supplies count uncertainty. Winner/nonwinner strata of the joint
score distribution are weighted to match the **exact original log5 ML**.
Totals and margins come from that same distribution. There are no starter,
recent-form, bullpen, park or market-price adjustments. This additional score
distribution assumption is explicit: totals are not uniquely determined by log5.

The run-line model beats this baseline in the small holdout; the total does not.
These fixed-threshold diagnostics still are not a historical price backtest.

## 2. Exactly one K-model change attempted

V1 excess Brier loss over baseline was 0.02704 among 14 pregame short-workload
profiles versus 0.00264 among 95 ordinary profiles. The one candidate removes
the additional last-five BF workload mixture when expected IP is below 4.5.
Expected strikeouts, season/recent shrinkage, lineup K adjustment, and residual
dispersion stay unchanged. This tests whether a second variance layer hurts
already-low-workload forecasts; no grid search or simultaneous parameter changes.

| Same 109 starts | Brier |
|---|---:|
| V1 | 0.247371 |
| Short-start candidate | 0.247706 |
| Season-rate baseline | 0.241593 |

**Reject the candidate; retain the original K model, informational only.**
Its larger-sample Brier comparison below is encouraging but calibration remains
worse than the simpler baseline, and retrospective conditional-lineup results
do not establish prospective betting returns. The candidate is retained only
as a reproducible optional research parameter; production defaults are unchanged.
The 109-start set was used for diagnosis, so its candidate result is exploratory,
not an untouched confirmation set.

## 3. Largest reconstructable existing 2026 window

Official schedule: **2,343 final games before September 22**. Actual starter logs
support **2,342 games**; one lacks a verifiable actual starter. **3,738 starts**
have actual K outcomes, opposing lineups, and at least three prior starts;
946 cold-start profiles are excluded under the unchanged v1 K eligibility rule.
Season data are batched, not fetched once per game: expansion used 27 source
calls and reused cached official inputs. No paid odds requests were made.

| Diagnostic | N | V1 model Brier | Baseline Brier | Model ECE |
|---|---:|---:|---:|---:|
| ML | 2,342 | 0.246289 | 0.249081 | 0.027459 |
| Home -1.5 | 2,342 | 0.229451 | 0.231811 | 0.036395 |
| Total over 8.5 | 2,342 | 0.251483 | 0.254752 | 0.042818 |
| K over 4.5 | 3,738 | 0.224580 | 0.227170 | 0.039682 |

K candidate Brier is **0.224616**, still slightly worse than unchanged v1.
Outside the diagnostic 109 starts: v1 **0.223895**, candidate **0.223922**,
baseline **0.226737** (3,629 starts). Baseline K ECE is **0.022907** on the
full sample, better than the model's calibration error despite worse Brier.

The small ML result was optimistic; expanded ML Brier rises to 0.2463. Totals
Brier is nearly unchanged, although the relative baseline ranking reverses.
Run-line and K Brier improve with the broader sample. These are descriptive
changes across different game mixes, not proof that the method became better
or a paired significance claim. Expanded data overlap the original holdout;
the older observations are additional chronological reconstruction, not future
post-selection validation.

Player archive: **2,306 immutable graded player-games, September 13–21**.
The unchanged HR interpolator is trained only on earlier dates and evaluated
walk-forward wherever its existing 25-observation bucket minimum is met:
**1,950 forecasts**, Brier **0.092423**, ECE **0.013654**. This is close to v1's
0.093432 Brier; the expanded evaluation uses rolling prior-day training rather
than a single fixed half-split.

Hits >=1: **55.594%** across 2,306 observations; HRR >=2: **41.544%**.
Both outcome rates remain close to v1's 55.556%/41.624%. **These are outcome
rates, not calibration.** Historical hit_gated and hrr_proj probabilities were
not frozen, so no larger honest Brier/calibration result exists for them. New
forecast archives fix this prospectively, not retroactively.

All performance inputs exclude the test game's entire date. Actual historical
starter/lineup identities condition these replays; announcement times, historical
park/weather and bullpen availability were not reconstructed. The backtest
does not silently substitute today's features for historical ones.

## 4. Repeated outage verification and real fixes

Run: `python -m unittest discover -s tests -p test_opportunity_followup.py -v`.

Executed the real refresh entry point three times with an injected HTTP 502,
plus a three-run 12:00/14:00/16:00 merge/freeze replay. Original game prices and
timestamps survive each persisted JSON write; STALE age increases from 2 to 6
hours; props remain FRESH; new captures contain no stale game contracts.
Recovery replaces the stale source with newly observed prices. Also tested:
props-only outage with healthy games, cold start with both sources unavailable,
stale/expired/after-kickoff capture rejection, and settlement of valid frozen bets.
This is deterministic fault injection, not a claim that three live scheduled
provider outages were observed.

**V1 needed a real fix:** props failures returned before attempting games, and
marked both sources stale. Fetch/parse failures now preserve and label the failed
source while the other source independently refreshes. Missing prices stay
missing; an outage cannot create fresh HR/hit/HRR prices. Those opportunities
are retained as STALE, not erased or promoted as current. Source timestamps,
failure reason and per-offer quote age are explicit in JSON.

Additional boundary hardening: both freezing and grading require a fresh,
verifiable pregame capture with a quote within the existing three-hour capture
window. Stale captures are ineligible/deferred. A quote that was genuinely fresh
when frozen remains valid historical evidence and **must still settle** even
if today's market feed is down; otherwise outages could suppress real losses.

## 5. Exploratory flat-stake ROI

Policy fixed before settled outcomes: **$1 per distinct exact selection**, first
pregame observation only, best book among observations at that same instant,
modeled conditional win probability at least **3 percentage points above
1/quoted decimal odds**, and positive modeled expected return. This offered-price
break-even rule works for one-sided HR quotes without inventing an Under price.
It is not the separate devigged-market comparison used elsewhere in the app.
No later best price is retrospectively substituted. Pushes return the stake;
voids are excluded. Realized edge means average win indicator minus offered-price
break-even probability on decisive bets. All metrics remain exploratory.

| Family | Qualifying frozen selections | Pending | Settled | Win rate / realized edge / ROI |
|---|---:|---:|---:|---|
| Game lines | 40 | 40 | 0 | Unavailable |
| Pitcher K | 16 | 16 | 0 | Unavailable |
| HR | 24 | 24 | 0 | Unavailable |
| Hits | 105 | 105 | 0 | Unavailable |
| HRR | 76 | 76 | 0 | Unavailable |

Genuine pregame offered-price records begin **September 22**; none has an
official final outcome yet. Earlier diagnostic holdouts contain outcomes and
reconstructable features, not frozen offered prices. Therefore historical ROI
cannot honestly be computed from them. Undefined values are JSON null, **not
0% ROI**. No -110 assumption, today's price or synthetic closing line is used.

The existing **Grade HR Board** workflow now recomputes and commits this ROI
summary after official grading (11:00/17:00/23:00 UTC plus the existing offseason
Monday retry). Future settled results populate each family automatically with
no browser or user-machine dependency. The hypothetical policy is not a record
of actual bets placed or a recommendation to stake on provisional models.
