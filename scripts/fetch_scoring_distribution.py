"""
Pull season team-game box scores and estimate pooled 1-, 2-, and 3-point
scoring-event frequencies.

The script writes:
1. a cleaned team-game box score table for the selected season/sample
2. a pooled scoring-distribution summary with counts and percentages

Example:
    conda run -n simapp-project python scripts/fetch_scoring_distribution.py \
        --season 2024-25 \
        --season-type "Regular Season"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd


def fetch_team_box_scores(
    season: str,
    season_type: str,
    game_id: str | None,
    date_from: str | None,
    date_to: str | None,
    timeout: float,
    retries: int,
    retry_backoff: float,
) -> pd.DataFrame:
    last_error = None
    for attempt in range(retries + 1):
        try:
            endpoint = leaguegamefinder.LeagueGameFinder(
                player_or_team_abbreviation="T",
                league_id_nullable="00",
                season_nullable=season,
                season_type_nullable=season_type,
                game_id_nullable=game_id,
                date_from_nullable=date_from,
                date_to_nullable=date_to,
                timeout=timeout,
            )
            box_scores = endpoint.get_data_frames()[0]
            break
        except Exception as error:
            last_error = error
            if attempt == retries:
                raise RuntimeError("Failed to fetch LeagueGameFinder data.") from last_error
            sleep_seconds = retry_backoff * (attempt + 1)
            print(
                f"LeagueGameFinder request failed "
                f"({error.__class__.__name__}: {error}). "
                f"Retrying in {sleep_seconds:.1f}s...",
                flush=True,
            )
            time.sleep(sleep_seconds)

    columns = [
        "GAME_ID",
        "GAME_DATE",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "TEAM_NAME",
        "MATCHUP",
        "WL",
        "MIN",
        "PTS",
        "FGM",
        "FGA",
        "FG3M",
        "FG3A",
        "FTM",
        "FTA",
        "PLUS_MINUS",
    ]
    box_scores = (
        box_scores.loc[:, columns]
        .drop_duplicates(subset=["GAME_ID", "TEAM_ID"])
        .sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"])
        .reset_index(drop=True)
    )

    numeric_columns = ["MIN", "PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "PLUS_MINUS"]
    for column in numeric_columns:
        box_scores[column] = pd.to_numeric(box_scores[column], errors="coerce").fillna(0)

    box_scores["one_point_makes"] = box_scores["FTM"].astype(int)
    box_scores["three_point_makes"] = box_scores["FG3M"].astype(int)
    box_scores["two_point_makes"] = (box_scores["FGM"] - box_scores["FG3M"]).astype(int)
    box_scores["total_scoring_events"] = (
        box_scores["one_point_makes"]
        + box_scores["two_point_makes"]
        + box_scores["three_point_makes"]
    )

    return box_scores


def subset_games(box_scores: pd.DataFrame, max_games: int | None) -> pd.DataFrame:
    if max_games is None:
        return box_scores

    selected_game_ids = (
        box_scores.loc[:, ["GAME_ID", "GAME_DATE"]]
        .drop_duplicates(subset=["GAME_ID"])
        .sort_values(["GAME_DATE", "GAME_ID"])
        .head(max_games)["GAME_ID"]
        .tolist()
    )
    return box_scores.loc[box_scores["GAME_ID"].isin(selected_game_ids)].copy()


def build_distribution(box_scores: pd.DataFrame, season: str, season_type: str) -> pd.DataFrame:
    counts = pd.DataFrame(
        {
            "points_scored": [1, 2, 3],
            "score_type": ["free_throw", "two_pointer", "three_pointer"],
            "count": [
                int(box_scores["one_point_makes"].sum()),
                int(box_scores["two_point_makes"].sum()),
                int(box_scores["three_point_makes"].sum()),
            ],
        }
    )
    total_events = int(counts["count"].sum())
    counts["percentage"] = counts["count"] / total_events if total_events else 0.0
    counts["season"] = season
    counts["season_type"] = season_type
    return counts.loc[:, ["season", "season_type", "points_scored", "score_type", "count", "percentage"]]


def write_outputs(
    box_scores: pd.DataFrame,
    distribution: pd.DataFrame,
    output_dir: Path,
    season: str,
    season_type: str,
):
    slug = f"{season}_{season_type.lower().replace(' ', '_')}"
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    box_scores.to_csv(processed_dir / f"team_game_box_scores_{slug}.csv", index=False)
    distribution.to_csv(processed_dir / f"scoring_distribution_{slug}.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch season team-game box scores and score distributions.")
    parser.add_argument("--season", required=True, help="Season string, e.g. 2024-25")
    parser.add_argument(
        "--season-type",
        default="Regular Season",
        help='Season type, e.g. "Regular Season" or "Playoffs"',
    )
    parser.add_argument(
        "--game-id",
        default=None,
        help="Optional single NBA game id to pull, e.g. 0022400001",
    )
    parser.add_argument(
        "--date-from",
        default=None,
        help="Optional start date filter in MM/DD/YYYY format.",
    )
    parser.add_argument(
        "--date-to",
        default=None,
        help="Optional end date filter in MM/DD/YYYY format.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Optional cap on the number of games to keep after the season pull.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Number of retries for failed requests.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=2.0,
        help="Linear backoff multiplier in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where processed CSVs will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    print(
        f"Pulling team-game box scores for {args.season} ({args.season_type})",
        flush=True,
    )
    if args.game_id is not None:
        print(f"Filtering to game id {args.game_id}.", flush=True)
    if args.date_from is not None or args.date_to is not None:
        print(
            f"Filtering to date range {args.date_from or 'start'} -> {args.date_to or 'end'}.",
            flush=True,
        )
    print("Requesting season box score data from LeagueGameFinder...", flush=True)
    box_scores = fetch_team_box_scores(
        season=args.season,
        season_type=args.season_type,
        game_id=args.game_id,
        date_from=args.date_from,
        date_to=args.date_to,
        timeout=args.timeout,
        retries=args.retries,
        retry_backoff=args.retry_backoff,
    )
    print(f"Fetched {len(box_scores)} team-game rows.", flush=True)

    if args.max_games is not None:
        print(f"Restricting sample to the first {args.max_games} games.", flush=True)
    box_scores = subset_games(box_scores, args.max_games)

    print("Building pooled scoring distribution...", flush=True)
    distribution = build_distribution(box_scores, args.season, args.season_type)

    print(f"Writing outputs to {output_dir.resolve()} ...", flush=True)
    write_outputs(box_scores, distribution, output_dir, args.season, args.season_type)

    print(f"Saved {len(box_scores)} team-game rows.", flush=True)
    print("\nPooled scoring distribution")
    print(distribution.to_string(index=False))


if __name__ == "__main__":
    main()
