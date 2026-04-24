"""
Parsimonious calibration helpers for the basketball prediction contract model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from MonteCarlo import MonteCarlo


class Calibration:
    """
    Calibrate team scoring intensities under a common mark distribution.

    The calibration uses:
    - a fixed pooled mark distribution p = [p1, p2, p3]
    - a market Team A win probability
    - a market total-points target

    It then backs out lambda_A and lambda_B.
    """

    def __init__(
        self,
        p,
        horizon,
        market_yes_prob,
        market_total_points,
        num_simulations=100_000,
        seed=42,
    ):
        self.p = np.asarray(p, dtype=float)
        self.horizon = float(horizon)
        self.market_yes_prob = float(market_yes_prob)
        self.market_total_points = float(market_total_points)
        self.num_simulations = int(num_simulations)
        self.seed = seed

        self.weights = np.array([1.0, 2.0, 3.0])
        self.expected_points_per_event = float(np.dot(self.weights, self.p))
        self.lambda_sum = self.market_total_points / (self.horizon * self.expected_points_per_event)

        self.scan = None
        self.lambda_a_hat = None
        self.lambda_b_hat = None
        self.solve_status = None
        self.final_results = None

    @classmethod
    def from_distribution_csv(
        cls,
        distribution_path,
        horizon,
        market_yes_prob,
        market_total_points,
        num_simulations=100_000,
        seed=42,
    ):
        distribution = pd.read_csv(distribution_path).sort_values("points_scored").reset_index(drop=True)
        p = distribution["percentage"].to_numpy(dtype=float)
        return cls(
            p=p,
            horizon=horizon,
            market_yes_prob=market_yes_prob,
            market_total_points=market_total_points,
            num_simulations=num_simulations,
            seed=seed,
        )

    def _contract(self, lambda_a):
        lambda_b = self.lambda_sum - lambda_a
        return {
            "lambda_a": float(lambda_a),
            "lambda_b": float(lambda_b),
            "p_a": self.p,
            "p_b": self.p,
            "T": float(self.horizon),
            "contract_type": "team_a_yes",
        }

    def model_yes_probability(self, lambda_a):
        pricer = MonteCarlo(
            self._contract(lambda_a),
            num_simulations=self.num_simulations,
            seed=self.seed,
        )
        return pricer.price()["price"]

    def objective(self, lambda_a):
        return self.model_yes_probability(lambda_a) - self.market_yes_prob

    def run_scan(self, num_grid=31):
        grid = np.linspace(
            max(1e-4, 0.01 * self.lambda_sum),
            self.lambda_sum - max(1e-4, 0.01 * self.lambda_sum),
            int(num_grid),
        )
        grid_probs = np.array([self.model_yes_probability(value) for value in grid])
        grid_objective = grid_probs - self.market_yes_prob

        self.scan = pd.DataFrame(
            {
                "lambda_a": grid,
                "lambda_b": self.lambda_sum - grid,
                "model_yes_prob": grid_probs,
                "objective": grid_objective,
            }
        )
        return self.scan.copy()

    def find_bracket(self):
        if self.scan is None:
            self.run_scan()

        lambda_a = self.scan["lambda_a"].to_numpy()
        objective = self.scan["objective"].to_numpy()

        for left, right, y_left, y_right in zip(lambda_a[:-1], lambda_a[1:], objective[:-1], objective[1:]):
            if y_left == 0:
                return float(left), float(left)
            if y_left * y_right < 0:
                return float(left), float(right)

        idx = int(np.argmin(np.abs(objective)))
        return float(lambda_a[idx]), float(lambda_a[idx])

    def calibrate(self, num_grid=31):
        self.run_scan(num_grid=num_grid)
        left, right = self.find_bracket()

        if left == right:
            self.lambda_a_hat = left
            self.solve_status = "No sign change found on the scan grid. Using the closest grid point."
        else:
            self.lambda_a_hat = float(brentq(self.objective, left, right))
            self.solve_status = "Root found with brentq on the scanned bracket."

        self.lambda_b_hat = float(self.lambda_sum - self.lambda_a_hat)
        final_pricer = MonteCarlo(
            self._contract(self.lambda_a_hat),
            num_simulations=self.num_simulations,
            seed=self.seed,
        )
        self.final_results = final_pricer.price()

        return {
            "lambda_sum": self.lambda_sum,
            "lambda_a_hat": self.lambda_a_hat,
            "lambda_b_hat": self.lambda_b_hat,
            "expected_points_per_event": self.expected_points_per_event,
            "model_yes_prob": self.final_results["price"],
            "model_yes_ci_low": self.final_results["confidence_interval"][0],
            "model_yes_ci_high": self.final_results["confidence_interval"][1],
            "solve_status": self.solve_status,
        }

    def summary(self, team_a="Team A", team_b="Team B"):
        if self.final_results is None:
            self.calibrate()

        return pd.Series(
            {
                "team_a": team_a,
                "team_b": team_b,
                "T_minutes": self.horizon,
                "market_yes_prob": self.market_yes_prob,
                "market_total_points": self.market_total_points,
                "expected_points_per_event": self.expected_points_per_event,
                "lambda_sum": self.lambda_sum,
                "lambda_a_hat": self.lambda_a_hat,
                "lambda_b_hat": self.lambda_b_hat,
                "model_yes_prob": self.final_results["price"],
                "model_yes_ci_low": self.final_results["confidence_interval"][0],
                "model_yes_ci_high": self.final_results["confidence_interval"][1],
                "solve_status": self.solve_status,
            }
        )


def load_distribution(distribution_path):
    distribution = pd.read_csv(Path(distribution_path)).sort_values("points_scored").reset_index(drop=True)
    p = distribution["percentage"].to_numpy(dtype=float)
    return distribution, p


def imply_intensities_from_spread_total(
    p,
    horizon,
    market_total_points,
    market_team_a_minus_team_b_spread,
    team_a="Team A",
    team_b="Team B",
):
    """
    Deterministically infer pregame scoring intensities from spread and total.
    """

    p = np.asarray(p, dtype=float)
    horizon = float(horizon)
    market_total_points = float(market_total_points)
    market_team_a_minus_team_b_spread = float(market_team_a_minus_team_b_spread)

    weights = np.array([1.0, 2.0, 3.0])
    expected_points_per_event = float(np.dot(weights, p))

    expected_points_a = 0.5 * (market_total_points + market_team_a_minus_team_b_spread)
    expected_points_b = 0.5 * (market_total_points - market_team_a_minus_team_b_spread)

    lambda_a_0 = expected_points_a / (horizon * expected_points_per_event)
    lambda_b_0 = expected_points_b / (horizon * expected_points_per_event)

    return {
        "team_a": team_a,
        "team_b": team_b,
        "T_minutes": horizon,
        "market_total_points": market_total_points,
        "market_team_a_minus_team_b_spread": market_team_a_minus_team_b_spread,
        "expected_points_per_event": expected_points_per_event,
        "expected_points_a": expected_points_a,
        "expected_points_b": expected_points_b,
        "lambda_a_0": lambda_a_0,
        "lambda_b_0": lambda_b_0,
        "lambda_total_0": lambda_a_0 + lambda_b_0,
        "strength_share_a_0": lambda_a_0 / (lambda_a_0 + lambda_b_0),
    }


def batch_imply_intensities_from_spread_total(
    market_data,
    p=None,
    distribution_path=None,
    team_a_col="team_a",
    team_b_col="team_b",
    horizon_col="T_minutes",
    market_total_col="market_total_points",
    spread_col="market_team_a_minus_team_b_spread",
):
    """
    Apply spread-plus-total implied intensity calibration across a market table.
    """

    if p is None:
        _, p = load_distribution(distribution_path)

    records = []

    for row_index, row in market_data.iterrows():
        team_a = row[team_a_col] if team_a_col in row.index else "Team A"
        team_b = row[team_b_col] if team_b_col in row.index else "Team B"

        result = imply_intensities_from_spread_total(
            p=p,
            horizon=row[horizon_col],
            market_total_points=row[market_total_col],
            market_team_a_minus_team_b_spread=row[spread_col],
            team_a=team_a,
            team_b=team_b,
        )
        result["row_index"] = row_index
        records.append(result)

    results = pd.DataFrame(records).set_index("row_index")
    results = results[[column for column in results.columns if column not in market_data.columns]]
    return market_data.join(results, how="left")


def calibrate_market(
    p,
    horizon,
    market_yes_prob,
    market_total_points,
    num_simulations=100_000,
    seed=42,
    num_grid=31,
    team_a="Team A",
    team_b="Team B",
):
    """
    Calibrate a single market observation and return a flat result dictionary.

    This is the lightweight parameter-driven entry point for scripts and
    backtests. It wraps the ``Calibration`` class and returns a single record
    that can be stored directly in a DataFrame.
    """

    calibration = Calibration(
        p=p,
        horizon=horizon,
        market_yes_prob=market_yes_prob,
        market_total_points=market_total_points,
        num_simulations=num_simulations,
        seed=seed,
    )
    calibration.calibrate(num_grid=num_grid)
    return calibration.summary(team_a=team_a, team_b=team_b).to_dict()


def backtest_markets(
    market_data,
    p=None,
    distribution_path=None,
    num_simulations=100_000,
    seed=42,
    num_grid=31,
    team_a_col="team_a",
    team_b_col="team_b",
    horizon_col="T",
    market_yes_col="market_yes_prob",
    market_total_col="market_total_points",
):
    """
    Run the parsimonious calibration across many market observations.

    Each row in ``market_data`` should contain:
    - a Team A win probability
    - a market total-points target
    - a game horizon in minutes

    Team name columns are optional; if they are absent, the generic Team A /
    Team B labels are used.
    """

    if p is None:
        _, p = load_distribution(distribution_path)

    records = []

    for row_index, row in market_data.iterrows():
        team_a = row[team_a_col] if team_a_col in row.index else "Team A"
        team_b = row[team_b_col] if team_b_col in row.index else "Team B"

        result = calibrate_market(
            p=p,
            horizon=row[horizon_col],
            market_yes_prob=row[market_yes_col],
            market_total_points=row[market_total_col],
            num_simulations=num_simulations,
            seed=seed,
            num_grid=num_grid,
            team_a=team_a,
            team_b=team_b,
        )
        result["row_index"] = row_index

        records.append(result)

    results = pd.DataFrame(records).set_index("row_index")
    results = results[[column for column in results.columns if column not in market_data.columns]]
    return market_data.join(results, how="left")
