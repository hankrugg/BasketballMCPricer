# Simulation Application Project

Monte Carlo pricer for digital prediction contracts using modular arrival processes and mark distributions.

## Environment

Create the Conda environment from the project root:

```bash
conda env create -f environment.yml
conda activate simapp-project
```

If you update dependencies later:

```bash
conda env update -f environment.yml --prune
```

## TODO

Finish proposal writeup

## Current Modules

- `src/MonteCarlo.py`
  - Pregame winner pricing under the baseline model
  - Live repricing from an in-game score state with `price_from_state(...)`
  - Pace updating with `update_total_pace(...)`

- `src/Calibration.py`
  - Parsimonious calibration from winner + total for notebook work
  - Deterministic sportsbook calibration from spread + total with
    `imply_intensities_from_spread_total(...)`
  - Batch implied-intensity calibration for market tables

- `src/Backtest.py`
  - Score-state alignment to Kalshi minute bars
  - Live fair-value generation through the game
  - Mark-to-market trading backtest against Kalshi bid/ask

## Data Scripts

- `scripts/fetch_scoring_distribution.py`
  - Pull NBA team-game box scores and estimate pooled historical
    `p1 / p2 / p3`

- `scripts/fetch_game_score_path.py`
  - Pull one NBA game's play-by-play
  - Reconstruct a scoring-event path with timestamps, elapsed minutes,
    cumulative score, and scoring-event counts
  - Write the one-game score path in the format expected by the backtest code

- `scripts/fetch_kalshi_historical_markets.py`
  - Pull historical Kalshi market metadata by ticker, event ticker, or series ticker
  - Optionally pull historical candlesticks over a requested time range
  - Write raw JSON and flattened CSVs under `data/`

- `scripts/find_kalshi_events.py`
  - Discover Kalshi event tickers and associated market tickers for a series
  - Filter to a season/date window and optional team-name text matches
  - Resolve market tickers from live nested markets or historical markets as needed

## Tests

Run the synthetic regression tests from the project root:

```bash
pytest -q
```
