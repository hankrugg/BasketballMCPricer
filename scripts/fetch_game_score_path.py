"""
Fetch a one-game NBA scoring-event path in the format used by the backtest code.

The script resolves an NBA game either from a supplied game id or from a date
plus two team abbreviations, downloads the play-by-play, and writes a processed
CSV containing:

- game_id
- timestamp
- elapsed_minutes
- score_a
- score_b
- total_scoring_events

The ``team_a`` / ``team_b`` orientation is controlled by the supplied team
abbreviations and should match the contract orientation used elsewhere.

Examples:
    conda run -n simapp-project python scripts/fetch_game_score_path.py \
        --date 2026-04-22 \
        --team-a-abbrev OKC \
        --team-b-abbrev PHX \
        --season-type Playoffs

    conda run -n simapp-project python scripts/fetch_game_score_path.py \
        --game-id 0042500172 \
        --team-a-abbrev OKC \
        --team-b-abbrev PHX
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from nba_api.live.nba.endpoints import playbyplay as live_playbyplay
from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd


def fetch_endpoint_with_retries(endpoint_factory, retries: int, retry_backoff: float):
    last_error = None
    for attempt in range(retries + 1):
        try:
            endpoint = endpoint_factory()
            return endpoint.get_data_frames()[0]
        except Exception as error:
            last_error = error
            if attempt == retries:
                raise RuntimeError("Failed to fetch NBA endpoint data.") from last_error
            sleep_seconds = retry_backoff * (attempt + 1)
            print(
                f"NBA request failed "
                f"({error.__class__.__name__}: {error}). "
                f"Retrying in {sleep_seconds:.1f}s...",
                flush=True,
            )
            time.sleep(sleep_seconds)


def fetch_live_actions_with_retries(endpoint_factory, retries: int, retry_backoff: float) -> pd.DataFrame:
    last_error = None
    for attempt in range(retries + 1):
        try:
            endpoint = endpoint_factory()
            payload = endpoint.get_dict()
            actions = payload.get("game", {}).get("actions", [])
            if not actions:
                raise RuntimeError("The play-by-play response did not include any actions.")
            return pd.DataFrame(actions)
        except Exception as error:
            last_error = error
            if attempt == retries:
                raise RuntimeError("Failed to fetch NBA live play-by-play data.") from last_error
            sleep_seconds = retry_backoff * (attempt + 1)
            print(
                f"NBA request failed "
                f"({error.__class__.__name__}: {error}). "
                f"Retrying in {sleep_seconds:.1f}s...",
                flush=True,
            )
            time.sleep(sleep_seconds)


def infer_season_from_date(game_date: pd.Timestamp) -> str:
    year = game_date.year
    if game_date.month >= 7:
        start_year = year
        end_year = year + 1
    else:
        start_year = year - 1
        end_year = year
    return f"{start_year}-{str(end_year)[-2:]}"


def fetch_game_rows(
    game_id: str | None,
    game_date: pd.Timestamp | None,
    season_type: str,
    season: str | None,
    timeout: float,
    retries: int,
    retry_backoff: float,
) -> pd.DataFrame:
    date_from = ""
    date_to = ""
    inferred_season = season or ""
    if game_date is not None:
        date_text = game_date.strftime("%m/%d/%Y")
        date_from = date_text
        date_to = date_text
        if not inferred_season:
            inferred_season = infer_season_from_date(game_date)

    return fetch_endpoint_with_retries(
        lambda: leaguegamefinder.LeagueGameFinder(
            player_or_team_abbreviation="T",
            league_id_nullable="00",
            game_id_nullable=game_id or "",
            date_from_nullable=date_from,
            date_to_nullable=date_to,
            season_type_nullable=season_type,
            season_nullable=inferred_season,
            timeout=timeout,
        ),
        retries=retries,
        retry_backoff=retry_backoff,
    )


def resolve_game_rows(
    game_rows: pd.DataFrame,
    game_id: str | None,
    team_a_abbrev: str,
    team_b_abbrev: str,
) -> pd.DataFrame:
    game_rows = game_rows.copy()
    game_rows["TEAM_ABBREVIATION"] = game_rows["TEAM_ABBREVIATION"].astype(str).str.upper()

    if game_id:
        resolved = game_rows.loc[game_rows["GAME_ID"].astype(str) == str(game_id)].copy()
    else:
        target_teams = {team_a_abbrev.upper(), team_b_abbrev.upper()}
        filtered = game_rows.loc[game_rows["TEAM_ABBREVIATION"].isin(target_teams)].copy()
        candidate_games = (
            filtered.groupby("GAME_ID")["TEAM_ABBREVIATION"]
            .agg(lambda values: set(values.astype(str)))
            .reset_index()
        )
        candidate_games = candidate_games.loc[candidate_games["TEAM_ABBREVIATION"].apply(lambda values: values == target_teams)]

        if candidate_games.empty:
            raise RuntimeError("No unique game matched the supplied teams and date.")
        if len(candidate_games) > 1:
            raise RuntimeError("Multiple games matched the supplied teams and date.")

        resolved_game_id = candidate_games.iloc[0]["GAME_ID"]
        resolved = filtered.loc[filtered["GAME_ID"] == resolved_game_id].copy()

    resolved = (
        resolved.drop_duplicates(subset=["GAME_ID", "TEAM_ID"])
        .sort_values(["GAME_ID", "TEAM_ID"])
        .reset_index(drop=True)
    )

    if resolved["GAME_ID"].nunique() != 1 or len(resolved) != 2:
        raise RuntimeError("Could not resolve exactly one two-team NBA game.")

    return resolved


def determine_home_away(game_rows: pd.DataFrame):
    home_mask = game_rows["MATCHUP"].astype(str).str.contains("vs.", regex=False)
    away_mask = game_rows["MATCHUP"].astype(str).str.contains("@", regex=False)

    if home_mask.sum() != 1 or away_mask.sum() != 1:
        raise RuntimeError("Could not infer home and away teams from MATCHUP.")

    home_row = game_rows.loc[home_mask].iloc[0]
    away_row = game_rows.loc[away_mask].iloc[0]

    return {
        "game_id": str(home_row["GAME_ID"]),
        "game_date": pd.to_datetime(home_row["GAME_DATE"]),
        "home_abbrev": str(home_row["TEAM_ABBREVIATION"]).upper(),
        "away_abbrev": str(away_row["TEAM_ABBREVIATION"]).upper(),
        "home_points": int(home_row["PTS"]),
        "away_points": int(away_row["PTS"]),
    }


def fetch_play_by_play(game_id: str, timeout: float, retries: int, retry_backoff: float) -> pd.DataFrame:
    return fetch_live_actions_with_retries(
        lambda: live_playbyplay.PlayByPlay(
            game_id=game_id,
            timeout=timeout,
        ),
        retries=retries,
        retry_backoff=retry_backoff,
    )


def compute_elapsed_minutes(period: int, clock_text: str):
    match = re.fullmatch(r"PT(?:(\d+)M)?([0-9]+(?:\.[0-9]+)?)S", str(clock_text))
    if match is None:
        raise RuntimeError(f"Could not parse clock value: {clock_text}")

    minutes_text = match.group(1) or "0"
    seconds_text = match.group(2)
    remaining_minutes = int(minutes_text) + float(seconds_text) / 60.0

    if int(period) <= 4:
        return (int(period) - 1) * 12.0 + (12.0 - remaining_minutes)

    return 48.0 + (int(period) - 5) * 5.0 + (5.0 - remaining_minutes)


def build_score_path(
    play_by_play: pd.DataFrame,
    team_a_abbrev: str,
    team_b_abbrev: str,
    home_away_info: dict,
) -> pd.DataFrame:
    required_columns = [
        "actionNumber",
        "period",
        "clock",
        "timeActual",
        "scoreHome",
        "scoreAway",
        "pointsTotal",
        "actionType",
        "description",
    ]
    pbp = play_by_play.loc[:, required_columns].copy()
    pbp = pbp.sort_values("actionNumber").reset_index(drop=True)

    pbp["scoreHome"] = pd.to_numeric(pbp["scoreHome"], errors="coerce")
    pbp["scoreAway"] = pd.to_numeric(pbp["scoreAway"], errors="coerce")
    pbp["pointsTotal"] = pd.to_numeric(pbp["pointsTotal"], errors="coerce")
    pbp = pbp.loc[pbp["scoreHome"].notna() & pbp["scoreAway"].notna()].copy()
    pbp["score_tuple"] = list(zip(pbp["scoreHome"], pbp["scoreAway"]))
    pbp["score_changed"] = pbp["score_tuple"].ne(pbp["score_tuple"].shift())
    scoring_rows = pbp.loc[pbp["score_changed"] & (pbp["pointsTotal"] > 0)].copy().reset_index(drop=True)

    if scoring_rows.empty:
        raise RuntimeError("No scored events were found in the play-by-play.")

    scoring_rows["home_score"] = scoring_rows["scoreHome"].astype(int)
    scoring_rows["away_score"] = scoring_rows["scoreAway"].astype(int)

    team_a_abbrev = team_a_abbrev.upper()
    team_b_abbrev = team_b_abbrev.upper()

    if team_a_abbrev == home_away_info["home_abbrev"]:
        scoring_rows["score_a"] = scoring_rows["home_score"]
        scoring_rows["score_b"] = scoring_rows["away_score"]
    elif team_a_abbrev == home_away_info["away_abbrev"]:
        scoring_rows["score_a"] = scoring_rows["away_score"]
        scoring_rows["score_b"] = scoring_rows["home_score"]
    else:
        raise RuntimeError("team_a_abbrev does not match either the home or away team.")

    if team_b_abbrev == home_away_info["home_abbrev"]:
        expected_team_b = scoring_rows["home_score"]
    elif team_b_abbrev == home_away_info["away_abbrev"]:
        expected_team_b = scoring_rows["away_score"]
    else:
        raise RuntimeError("team_b_abbrev does not match either the home or away team.")

    if not (scoring_rows["score_b"].to_numpy() == expected_team_b.to_numpy()).all():
        raise RuntimeError("team_b_abbrev orientation does not match reconstructed score path.")

    scoring_rows["elapsed_minutes"] = scoring_rows.apply(
        lambda row: compute_elapsed_minutes(row["period"], row["clock"]),
        axis=1,
    )
    scoring_rows["timestamp"] = pd.to_datetime(scoring_rows["timeActual"], utc=True, errors="coerce")
    scoring_rows["game_id"] = home_away_info["game_id"]
    scoring_rows["team_a_abbrev"] = team_a_abbrev
    scoring_rows["team_b_abbrev"] = team_b_abbrev
    scoring_rows["total_scoring_events"] = range(1, len(scoring_rows) + 1)
    scoring_rows["current_margin"] = scoring_rows["score_a"] - scoring_rows["score_b"]

    return scoring_rows.loc[
        :,
        [
            "game_id",
            "timestamp",
            "elapsed_minutes",
            "score_a",
            "score_b",
            "total_scoring_events",
            "current_margin",
            "team_a_abbrev",
            "team_b_abbrev",
            "period",
            "clock",
            "timeActual",
            "actionType",
            "description",
        ],
    ]


def slugify(text: str):
    return text.lower().replace("/", "_").replace(" ", "_")


def write_outputs(score_path: pd.DataFrame, output_dir: Path, game_id: str, team_a_abbrev: str, team_b_abbrev: str):
    processed_dir = output_dir / "processed" / "nba"
    processed_dir.mkdir(parents=True, exist_ok=True)

    slug = f"{game_id}_{slugify(team_a_abbrev)}_{slugify(team_b_abbrev)}"
    output_path = processed_dir / f"score_path_{slug}.csv"
    score_path.to_csv(output_path, index=False)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch a one-game NBA score path for the backtest pipeline.")
    parser.add_argument("--game-id", default=None, help="Optional NBA game id.")
    parser.add_argument("--date", default=None, help="Game date in YYYY-MM-DD format when resolving by teams.")
    parser.add_argument("--team-a-abbrev", required=True, help="Team A abbreviation, e.g. OKC.")
    parser.add_argument("--team-b-abbrev", required=True, help="Team B abbreviation, e.g. PHX.")
    parser.add_argument(
        "--season-type",
        default="Playoffs",
        help='Season type, e.g. "Regular Season" or "Playoffs".',
    )
    parser.add_argument(
        "--season",
        default=None,
        help='Optional NBA season string like "2025-26". If omitted, it is inferred from --date.',
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Number of retries for failed NBA requests.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=2.0,
        help="Linear retry backoff in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where the processed score path CSV will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.game_id is None and args.date is None:
        raise SystemExit("Provide either --game-id or --date.")

    team_a_abbrev = args.team_a_abbrev.upper()
    team_b_abbrev = args.team_b_abbrev.upper()
    game_date = pd.Timestamp(args.date) if args.date is not None else None

    print("Requesting game rows from LeagueGameFinder...", flush=True)
    game_rows = fetch_game_rows(
        game_id=args.game_id,
        game_date=game_date,
        season_type=args.season_type,
        season=args.season,
        timeout=args.timeout,
        retries=args.retries,
        retry_backoff=args.retry_backoff,
    )

    resolved_game_rows = resolve_game_rows(
        game_rows=game_rows,
        game_id=args.game_id,
        team_a_abbrev=team_a_abbrev,
        team_b_abbrev=team_b_abbrev,
    )
    home_away_info = determine_home_away(resolved_game_rows)

    print(
        f"Resolved game {home_away_info['game_id']}: "
        f"{home_away_info['away_abbrev']} at {home_away_info['home_abbrev']}",
        flush=True,
    )
    print("Requesting play-by-play...", flush=True)
    play_by_play = fetch_play_by_play(
        game_id=home_away_info["game_id"],
        timeout=args.timeout,
        retries=args.retries,
        retry_backoff=args.retry_backoff,
    )

    print("Building scoring-event path...", flush=True)
    score_path = build_score_path(
        play_by_play=play_by_play,
        team_a_abbrev=team_a_abbrev,
        team_b_abbrev=team_b_abbrev,
        home_away_info=home_away_info,
    )

    output_path = write_outputs(
        score_path=score_path,
        output_dir=Path(args.output_dir),
        game_id=home_away_info["game_id"],
        team_a_abbrev=team_a_abbrev,
        team_b_abbrev=team_b_abbrev,
    )

    print(f"Saved {len(score_path)} scoring-event rows to {output_path}", flush=True)
    print("\nScore path preview")
    print(score_path.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
