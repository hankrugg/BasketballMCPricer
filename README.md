# Basketball Monte Carlo Prediction Market Study

This project reproduces a one-game study of the April 22, 2026 Phoenix Suns at
Oklahoma City Thunder playoff game. It compares Kalshi's OKC winner contract
price path with a simple Monte Carlo model-implied win probability path.

The cleaned project scope is intentionally narrow:

- Calibrate fixed pregame scoring intensities from market inputs.
- Align the observed score state to each Kalshi minute timestamp.
- Reprice the OKC winner contract by Monte Carlo from each in-game state.
- Compare the model path against the Kalshi price path.

The model does not update scoring intensities during the game, and there are no
backtest files in the current project.

## Environment

Create the Conda environment from the project root:

```bash
conda env create -f environment.yml
conda activate simapp-project
```

If dependencies change later:

```bash
conda env update -f environment.yml --prune
```

## Main Notebook

Run the one-game analysis here:

```text
notebooks/PHXOKC.ipynb
```

The notebook uses relative paths, so it can be opened from either the project
root or the `notebooks/` folder. It performs the full workflow:

1. Load the pooled NBA scoring distribution.
2. Load Kalshi minute-level OKC yes prices.
3. Load the reconstructed OKC/PHX score path.
4. Convert timestamps from UTC to Eastern time.
5. Calibrate pregame scoring intensities.
6. Carry forward the latest score state to each Kalshi timestamp.
7. Run Monte Carlo repricing at each timestamp.
8. Plot model-implied OKC win probability against Kalshi.

## Current Code

- `src/MonteCarlo.py`
  - `MonteCarlo`: simulates terminal scoring counts and prices a Team A winner contract.
  - `price_from_state(...)`: reprices from current margin and time remaining.

- `src/Calibration.py`
  - `Calibration`: solves team scoring intensities from Kalshi yes probability plus total.
  - `imply_intensities_from_spread_total(...)`: computes a spread-plus-total reference calibration.
  - `build_score_state_grid(...)`: aligns irregular NBA scoring events to Kalshi minute bars without future leakage.
  - `load_distribution(...)`: loads the pooled scoring mark distribution.

## Data

The notebook expects these processed files:

- `data/processed/scoring_distribution_2024-25_regular_season.csv`
- `data/processed/kalshi/kalshi-price-history-kxnbagame-26apr22phxokc-minute.csv`
- `data/processed/nba/score_path_0042500142_okc_phx.csv`

There is also a source team-game box score file used to produce the scoring
distribution:

- `data/processed/team_game_box_scores_2024-25_regular_season.csv`

## Data Scripts

- `scripts/fetch_scoring_distribution.py`
  - Pulls NBA team-game box scores.
  - Estimates the pooled probabilities for 1-, 2-, and 3-point scoring events.

- `scripts/fetch_game_score_path.py`
  - Pulls one NBA game's play-by-play.
  - Reconstructs scoring events with timestamps, elapsed minutes, cumulative score, and scoring-event counts.
  - Writes the score path format used by `notebooks/PHXOKC.ipynb`.

The Kalshi minute data is already stored in `data/processed/kalshi/` for this
project; no Kalshi-fetching script is currently included.

## Report

The short writeup is:

```text
tex/report.tex
```

To rebuild the PDF from the `tex/` folder:

```bash
pdflatex -interaction=nonstopmode report.tex
```

## Quick Verification

From the project root, this checks that the local modules import:

```bash
python -B -c "import sys; sys.path.insert(0, 'src'); import MonteCarlo, Calibration; print('imports ok')"
```
