"""
Live Kalshi edge backtest using pregame sportsbook priors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from Calibration import imply_intensities_from_spread_total
from MonteCarlo import price_from_state, update_total_pace


def build_score_state_grid(play_by_play_df, kalshi_bar_timestamps):
    """
    Carry forward the latest known score state to each Kalshi bar timestamp.
    """

    state_columns = ["timestamp", "elapsed_minutes", "score_a", "score_b", "total_scoring_events"]

    if isinstance(kalshi_bar_timestamps, pd.DataFrame):
        bars = kalshi_bar_timestamps.loc[:, ["timestamp"]].copy()
    elif isinstance(kalshi_bar_timestamps, pd.Series):
        bars = pd.DataFrame({"timestamp": kalshi_bar_timestamps})
    else:
        bars = pd.DataFrame({"timestamp": list(kalshi_bar_timestamps)})

    bars["timestamp"] = pd.to_datetime(bars["timestamp"])
    bars = bars.sort_values("timestamp").reset_index(drop=True)

    play_by_play = play_by_play_df.loc[:, state_columns].copy()
    play_by_play["timestamp"] = pd.to_datetime(play_by_play["timestamp"])
    play_by_play = (
        play_by_play.sort_values("timestamp")
        .groupby("timestamp", as_index=False)
        .last()
    )

    merged = pd.merge_asof(
        bars,
        play_by_play,
        on="timestamp",
        direction="backward",
    )

    merged["elapsed_minutes"] = merged["elapsed_minutes"].fillna(0.0).astype(float)
    merged["score_a"] = merged["score_a"].fillna(0).astype(int)
    merged["score_b"] = merged["score_b"].fillna(0).astype(int)
    merged["total_scoring_events"] = merged["total_scoring_events"].fillna(0).astype(int)
    merged["current_margin"] = merged["score_a"] - merged["score_b"]

    if play_by_play.empty:
        final_score_a = 0
        final_score_b = 0
        max_elapsed_minutes = 0.0
    else:
        final_score_a = int(play_by_play["score_a"].max())
        final_score_b = int(play_by_play["score_b"].max())
        max_elapsed_minutes = float(play_by_play["elapsed_minutes"].max())

    merged["final_score_a"] = final_score_a
    merged["final_score_b"] = final_score_b
    merged["max_elapsed_minutes"] = max_elapsed_minutes
    return merged


def _mark_to_market_value(position, yes_bid, yes_ask):
    if position == "long_yes":
        return float(yes_bid)
    if position == "long_no":
        return float(1.0 - yes_ask)
    return 0.0


def _game_summary_template(game_id, team_a, team_b, skipped=False, skip_reason=None):
    return {
        "game_id": game_id,
        "team_a": team_a,
        "team_b": team_b,
        "skipped": skipped,
        "skip_reason": skip_reason,
        "num_trades": 0,
        "average_entry_edge": np.nan,
        "realized_pnl": np.nan,
        "final_score_a": np.nan,
        "final_score_b": np.nan,
        "yes_settlement": np.nan,
    }


def backtest_game(
    kalshi_bars,
    score_state_grid,
    pregame_market_row,
    p,
    num_simulations=100_000,
    prior_weight_minutes=12.0,
    edge_threshold=0.03,
    seed=42,
):
    """
    Backtest one game on Kalshi minute bars using live model fair values.
    """

    market_row = pd.Series(pregame_market_row)
    game_id = market_row.get("game_id")
    team_a = market_row.get("team_a", "Team A")
    team_b = market_row.get("team_b", "Team B")
    T_minutes = float(market_row["T_minutes"])

    summary = _game_summary_template(game_id=game_id, team_a=team_a, team_b=team_b)

    if abs(T_minutes - 48.0) > 1e-9:
        summary.update(skipped=True, skip_reason="non_regulation_horizon")
        return {"signals": pd.DataFrame(), "trades": pd.DataFrame(), "summary": summary}

    if score_state_grid.empty:
        summary.update(skipped=True, skip_reason="missing_score_state")
        return {"signals": pd.DataFrame(), "trades": pd.DataFrame(), "summary": summary}

    if float(score_state_grid["max_elapsed_minutes"].max()) > T_minutes:
        summary.update(skipped=True, skip_reason="overtime_excluded")
        return {"signals": pd.DataFrame(), "trades": pd.DataFrame(), "summary": summary}

    implied = imply_intensities_from_spread_total(
        p=p,
        horizon=T_minutes,
        market_total_points=market_row["market_total_points"],
        market_team_a_minus_team_b_spread=market_row["market_team_a_minus_team_b_spread"],
        team_a=team_a,
        team_b=team_b,
    )

    bars = kalshi_bars.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"])
    bars = (
        bars.sort_values("timestamp")
        .dropna(subset=["yes_bid", "yes_ask"])
        .reset_index(drop=True)
    )

    if bars.empty:
        summary.update(skipped=True, skip_reason="missing_kalshi_quotes")
        return {"signals": pd.DataFrame(), "trades": pd.DataFrame(), "summary": summary}

    score_state = score_state_grid.copy()
    score_state["timestamp"] = pd.to_datetime(score_state["timestamp"])

    signals = bars.merge(
        score_state,
        on="timestamp",
        how="left",
        validate="one_to_one",
    )

    if signals.empty:
        summary.update(skipped=True, skip_reason="no_signal_rows")
        return {"signals": pd.DataFrame(), "trades": pd.DataFrame(), "summary": summary}

    lambda_total_0 = implied["lambda_total_0"]
    strength_share_a_0 = implied["strength_share_a_0"]

    position = "flat"
    cash = 0.0
    entry_edges = []
    trade_records = []
    signal_records = []

    for row_number, row in enumerate(signals.itertuples(index=False), start=1):
        current_margin = int(row.current_margin)
        elapsed_minutes = float(row.elapsed_minutes)
        time_remaining = max(T_minutes - elapsed_minutes, 0.0)

        lambda_total_t = update_total_pace(
            lambda_total_0=lambda_total_0,
            elapsed_minutes=elapsed_minutes,
            total_scoring_events_observed=row.total_scoring_events,
            prior_weight_minutes=prior_weight_minutes,
        )
        lambda_a_t = strength_share_a_0 * lambda_total_t
        lambda_b_t = (1.0 - strength_share_a_0) * lambda_total_t

        pricing = price_from_state(
            current_margin=current_margin,
            time_remaining=time_remaining,
            lambda_a=lambda_a_t,
            lambda_b=lambda_b_t,
            p=p,
            num_simulations=num_simulations,
            seed=seed + row_number,
        )
        fair_yes = pricing["price"]
        standard_error = pricing["standard_error"]
        ci_low, ci_high = pricing["confidence_interval"]

        yes_bid = float(row.yes_bid)
        yes_ask = float(row.yes_ask)
        no_bid = 1.0 - yes_ask
        no_ask = 1.0 - yes_bid

        buy_yes_edge = fair_yes - yes_ask
        buy_no_edge = yes_bid - fair_yes

        if position == "flat":
            if buy_yes_edge > edge_threshold and buy_yes_edge >= buy_no_edge:
                cash -= yes_ask
                position = "long_yes"
                entry_edges.append(buy_yes_edge)
                trade_records.append(
                    {
                        "game_id": game_id,
                        "timestamp": row.timestamp,
                        "action": "buy_yes",
                        "price": yes_ask,
                        "edge": buy_yes_edge,
                        "inventory_after": position,
                    }
                )
            elif buy_no_edge > edge_threshold:
                cash -= no_ask
                position = "long_no"
                entry_edges.append(buy_no_edge)
                trade_records.append(
                    {
                        "game_id": game_id,
                        "timestamp": row.timestamp,
                        "action": "buy_no",
                        "price": no_ask,
                        "edge": buy_no_edge,
                        "inventory_after": position,
                    }
                )
        elif position == "long_yes" and buy_no_edge > edge_threshold:
            cash += yes_bid
            trade_records.append(
                {
                    "game_id": game_id,
                    "timestamp": row.timestamp,
                    "action": "sell_yes",
                    "price": yes_bid,
                    "edge": buy_no_edge,
                    "inventory_after": "flat",
                }
            )
            cash -= no_ask
            position = "long_no"
            entry_edges.append(buy_no_edge)
            trade_records.append(
                {
                    "game_id": game_id,
                    "timestamp": row.timestamp,
                    "action": "buy_no",
                    "price": no_ask,
                    "edge": buy_no_edge,
                    "inventory_after": position,
                }
            )
        elif position == "long_no" and buy_yes_edge > edge_threshold:
            cash += no_bid
            trade_records.append(
                {
                    "game_id": game_id,
                    "timestamp": row.timestamp,
                    "action": "sell_no",
                    "price": no_bid,
                    "edge": buy_yes_edge,
                    "inventory_after": "flat",
                }
            )
            cash -= yes_ask
            position = "long_yes"
            entry_edges.append(buy_yes_edge)
            trade_records.append(
                {
                    "game_id": game_id,
                    "timestamp": row.timestamp,
                    "action": "buy_yes",
                    "price": yes_ask,
                    "edge": buy_yes_edge,
                    "inventory_after": position,
                }
            )

        mark_to_market_value = _mark_to_market_value(position, yes_bid=yes_bid, yes_ask=yes_ask)
        mark_to_market_pnl = cash + mark_to_market_value

        signal_records.append(
            {
                "game_id": game_id,
                "timestamp": row.timestamp,
                "team_a": team_a,
                "team_b": team_b,
                "score_a": int(row.score_a),
                "score_b": int(row.score_b),
                "total_scoring_events": int(row.total_scoring_events),
                "elapsed_minutes": elapsed_minutes,
                "time_remaining": time_remaining,
                "current_margin": current_margin,
                "lambda_total_t": lambda_total_t,
                "lambda_a_t": lambda_a_t,
                "lambda_b_t": lambda_b_t,
                "fair_yes": fair_yes,
                "standard_error": standard_error,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "buy_yes_edge": buy_yes_edge,
                "buy_no_edge": buy_no_edge,
                "position": position,
                "cash": cash,
                "mark_to_market_value": mark_to_market_value,
                "mark_to_market_pnl": mark_to_market_pnl,
            }
        )

    final_score_a = int(score_state_grid["final_score_a"].iloc[0])
    final_score_b = int(score_state_grid["final_score_b"].iloc[0])

    if final_score_a == final_score_b:
        summary.update(skipped=True, skip_reason="tied_regulation")
        return {
            "signals": pd.DataFrame(signal_records),
            "trades": pd.DataFrame(trade_records),
            "summary": summary,
        }

    yes_settlement = float(final_score_a > final_score_b)

    if position == "long_yes":
        cash += yes_settlement
        trade_records.append(
            {
                "game_id": game_id,
                "timestamp": pd.NaT,
                "action": "settle_yes",
                "price": yes_settlement,
                "edge": np.nan,
                "inventory_after": "flat",
            }
        )
    elif position == "long_no":
        cash += 1.0 - yes_settlement
        trade_records.append(
            {
                "game_id": game_id,
                "timestamp": pd.NaT,
                "action": "settle_no",
                "price": 1.0 - yes_settlement,
                "edge": np.nan,
                "inventory_after": "flat",
            }
        )

    summary.update(
        skipped=False,
        skip_reason=None,
        num_trades=sum(record["action"].startswith("buy") or record["action"].startswith("sell") for record in trade_records),
        average_entry_edge=float(np.mean(entry_edges)) if entry_edges else np.nan,
        realized_pnl=float(cash),
        final_score_a=final_score_a,
        final_score_b=final_score_b,
        yes_settlement=yes_settlement,
        lambda_a_0=implied["lambda_a_0"],
        lambda_b_0=implied["lambda_b_0"],
        lambda_total_0=implied["lambda_total_0"],
    )

    return {
        "signals": pd.DataFrame(signal_records),
        "trades": pd.DataFrame(trade_records),
        "summary": summary,
    }


def backtest_games(
    kalshi_bars_df,
    score_events_df,
    sportsbook_df,
    p,
    num_simulations=100_000,
    prior_weight_minutes=12.0,
    edge_threshold=0.03,
    seed=42,
):
    """
    Run the live backtest across many games.
    """

    game_summaries = []
    signal_tables = []
    trade_tables = []

    sportsbook_rows = sportsbook_df.copy()

    for game_number, market_row in enumerate(sportsbook_rows.itertuples(index=False), start=1):
        game_id = market_row.game_id
        kalshi_bars = kalshi_bars_df.loc[kalshi_bars_df["game_id"] == game_id].copy()
        score_events = score_events_df.loc[score_events_df["game_id"] == game_id].copy()

        if score_events.empty:
            game_summaries.append(
                _game_summary_template(
                    game_id=game_id,
                    team_a=getattr(market_row, "team_a", "Team A"),
                    team_b=getattr(market_row, "team_b", "Team B"),
                    skipped=True,
                    skip_reason="missing_score_events",
                )
            )
            continue

        score_state_grid = build_score_state_grid(score_events, kalshi_bars["timestamp"])
        result = backtest_game(
            kalshi_bars=kalshi_bars,
            score_state_grid=score_state_grid,
            pregame_market_row=pd.Series(market_row._asdict()),
            p=p,
            num_simulations=num_simulations,
            prior_weight_minutes=prior_weight_minutes,
            edge_threshold=edge_threshold,
            seed=seed + game_number * 1000,
        )

        game_summaries.append(result["summary"])

        if not result["signals"].empty:
            signal_tables.append(result["signals"])
        if not result["trades"].empty:
            trade_tables.append(result["trades"])

    game_summaries_df = pd.DataFrame(game_summaries)
    signals_df = pd.concat(signal_tables, ignore_index=True) if signal_tables else pd.DataFrame()
    trades_df = pd.concat(trade_tables, ignore_index=True) if trade_tables else pd.DataFrame()

    evaluated_games = game_summaries_df.loc[~game_summaries_df["skipped"]].copy()

    aggregate_summary = {
        "total_games": int(len(game_summaries_df)),
        "games_evaluated": int(len(evaluated_games)),
        "games_skipped": int(game_summaries_df["skipped"].sum()),
        "games_traded": int((evaluated_games["num_trades"] > 0).sum()) if not evaluated_games.empty else 0,
        "total_trades": int(evaluated_games["num_trades"].sum()) if not evaluated_games.empty else 0,
        "total_realized_pnl": float(evaluated_games["realized_pnl"].sum()) if not evaluated_games.empty else 0.0,
        "average_game_pnl": float(evaluated_games["realized_pnl"].mean()) if not evaluated_games.empty else np.nan,
        "average_entry_edge": float(evaluated_games["average_entry_edge"].mean()) if not evaluated_games.empty else np.nan,
    }

    return {
        "signals": signals_df,
        "trades": trades_df,
        "game_summaries": game_summaries_df,
        "aggregate_summary": aggregate_summary,
    }
