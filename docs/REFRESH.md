# Refresh schedule

The shared GOING Yard page checks live Parlay prices every five minutes while visible with upcoming games, and hourly without upcoming games. Requests do not overlap; returning to the tab triggers a due check. The existing board snapshot reload remains every five minutes. Snapshot and live odds caches on Vercel last two minutes.

Heavy board builds run at 10:30 UTC and every two hours at :17 from 14:17 through 02:17 UTC. A lightweight MLB schedule check looks back three days and ahead seven days. With no games nearby, heavy builds run only Mondays at 10:30 UTC and grading Mondays at 10:40 UTC. In-season grading retains three daily runs. Existing microclimate profiles refresh daily in season and Mondays at 10:50 UTC otherwise. Lightweight live HR updates run every 15 minutes from 16:00 through 04:45 UTC when games are nearby. Manual runs bypass the seasonal gate; a schedule outage preserves the normal cadence rather than silently disabling updates.

GitHub scheduled jobs can start late. Completed published snapshots are read directly from this repository by GOING; a Vercel rebuild is not required for each data refresh. No predictive calculations changed.
