# GTA VI Twitch CCU forecast

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (hours 2-24, UTC), as five quantiles per hour, from
the first hour's viewer count, the pre-launch history, and the release-day
curves of past big game launches.

## How it works

`main.py` reads one case file and:

1. Expresses every comparable launch's release-day curve as log-ratios to
   its own first hour.
2. Weights the launches by similarity to the target case, using only hourly
   data: first-hour level, pre-launch ramp, pre-midnight level, and
   pre-launch peak.
3. Takes similarity-weighted quantiles of the log-ratios per hour, applies
   them to the target's first-hour print, shrinks early hours toward
   persistence, and calibrates the spread with a floor.
4. Falls back to a persistence forecast with a widening spread whenever a
   case carries no usable comparables, so an output is always written.

Standard library only, no network, deterministic: two runs on the same case
write identical bytes.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

[![gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/apasi22/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/apasi22/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
