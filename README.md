# GTA VI trophy rarity forecast — week one

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/luthiiiialvaro/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/badge/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/luthiiiialvaro/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, its
earn rate on [Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
at 2026-11-26T00:00:00Z — as five quantiles (5%, 25%, 50%, 75%, 95%).
Entry for the make.fun bounty
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## The model

`main.py` (Python 3.13, standard library only, no network, no
randomness) reads a case file — the house's reads of the Exophase
trophy page before the checkpoint — and writes one forecast per case.
It never fetches Exophase; its only data is the case file's history.

Three ideas, in order of importance:

1. **A saturating trend per trophy, shrunk twice.** The logit of each
   trophy's earn rate is fitted against log time since launch with a
   straight line, weighted toward recent reads (8-hour half-life: only
   the current regime extrapolates). The slope is shrunk toward the
   median slope of the trophy's tier, so a rare trophy borrows strength
   from its peers. Then the extrapolated move is shrunk toward
   persistence — only 30% of the trend's move is taken — because a
   trend fitted on 2.5 days cannot be trusted to run untouched for
   4.5 more, and the pinball loss punishes an overshoot as much as an
   undershoot. A quadratic fit was tried and dropped: over a 4.5-day
   horizon its curvature estimate is too unstable, and on synthetic
   finals it sent forecasts to 100%.
2. **Tracked players, and the earners floor.** Log tracked players is
   fitted the same way (never allowed to shrink, capped at 8x the last
   read). Earners never decrease, so the forecast rate can never fall
   below 100 × E(last) / N(target). That floor is the dilution story
   in one line: the rate falls only as far as newcomers outnumber new
   earners.
3. **A stale-page branch.** Exophase refreshes on its own schedule, so
   a case whose whole history is one unchanged page (launch night:
   every read the same, 638 tracked) carries no trend at all. For that
   case the forecast assumes the next refresh: the tracked population
   jumps (5x the last read) and new players earn each trophy at a
   fifth of the launch-night cohort's rate, so every rate but the 100%
   one falls, with a wide spread. This branch only fires on a fully
   flat history, so a final case with real reads never takes it.

**Spread.** Quantiles are the median on the logit scale plus a normal
spread that grows with the horizon: a random-walk term (0.40 logit
units per root day), a slope-uncertainty term (30% of the fitted slope
times the extrapolation distance in log time), and a refresh-jump term
estimated from the history's own jumps between consecutive reads. A
trophy at 99% or above is special-cased near the top (median 99.5–99.9,
wide spread).

## How to run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` in the input directory, in name order, and writes
one `<id>.out` per case: `{"trophies": [{"id": n, "q": [q05, q25, q50,
q75, q95]}, ...]}` — five non-decreasing quantiles in percent. Runs in
about 0.1 s per case, deterministic (two runs write identical bytes).

## Validation

There is no local grader — the server grades. Locally, the program was
checked on the kit's practice cases (p00, p01) and on two synthetic
final-like cases built from the same dynamics (a 358-read history to the
deadline, rates following saturating curves to known ceilings, answers
at the target). The kit ships no practice answers, so the practice was
scored against several plausible answers built from the known refresh
facts (3,868 players tracked at the 04:30Z refresh, every rate but the
100% one lower). Mean pinball loss on the logit scale (lower is better):

| case | this model | persistence baseline |
| --- | --- | --- |
| practice, dilution f=0.20 | 0.221 | 0.996 |
| practice, dilution f=0.35 | 0.254 | 0.680 |
| practice, rates quartered | 0.275 | 1.265 |
| practice, rates halved | 0.271 | 0.593 |
| synthetic final, still climbing | 0.227 | 0.310 |
| synthetic final, saturating early | 0.212 | 0.224 |

The house baselines score about 1.51 on the practice; this model is
well below that on every plausible practice answer, with 90% coverage
of 0.98–1.00. The one answer it loses on is "no change at all", which
the rules themselves rule out (every rate fell at the refresh). The
server's own practice score after the first submit is the real
feedback: if it lands far above these local numbers, the flat-history
branch's dilution prior (5x tracked growth, new players earning at a
fifth of the old cohort's rate) is the first thing to revisit.

## Files

- `main.py` — the forecast program (the entry's only entrypoint).
- `LICENSE` — MIT.
- `.makefun/owner.json` — the entry wallet.
