# Roblox Up-and-Coming top-10 forecaster

Statistical forecaster for `roblox-up-and-coming-forecast-01` (make.fun).

Method: per game, fit rank velocity over the last 12h of reads, diffuse the
rank with a Gaussian random walk (sigma 1.0 ranks/sqrt(hour), momentum decay
tau 24h), average the rank distribution over horizons 0..72h. 28-day age
expiry is applied when a game's created date is known. Each place leaves an
unnamed share of 0.2 for churn and new entrants.

Practice backtest (train on the kit's 34 reads, score on the 10 archive reads
of the practice window): mean log score **-35.2** vs the house baseline's
**-43.51**.

Run: `MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py`
