"""
Monte Carlo pricer for basketball winner contracts under the baseline model.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


class MonteCarlo:
    """
    Monte Carlo pricer for a Team A winner contract.

    The contract is passed in as a dictionary with the fields
    ``lambda_a``, ``lambda_b``, ``p_a``, ``p_b``, and ``T``.
    The probability vectors ``p_a`` and ``p_b`` are ordered as
    ``[p_1, p_2, p_3]``.
    """

    def __init__(self, contract, num_simulations, seed=42):
        self.contract = contract
        self.num_simulations = int(num_simulations)
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.margins = None
        self.payoffs = None
        self.paths = None
        self.price_estimate = None
        self.std_error = None
        self.ci = None

    def _poisson_means(self):
        horizon = float(self.contract["T"])
        p_a = np.asarray(self.contract["p_a"], dtype=float)
        p_b = np.asarray(self.contract["p_b"], dtype=float)
        return {
            "team_a": float(self.contract["lambda_a"]) * p_a * horizon,
            "team_b": float(self.contract["lambda_b"]) * p_b * horizon,
        }

    def _confidence_interval(self, estimate, n_paths):
        standard_error = math.sqrt(estimate * (1.0 - estimate) / n_paths)
        ci_low = max(0.0, estimate - 1.96 * standard_error)
        ci_high = min(1.0, estimate + 1.96 * standard_error)
        return standard_error, (ci_low, ci_high)

    def _payoff(self, margins):
        return (margins > 0).astype(float)

    def _single_path(self):
        horizon = float(self.contract["T"])
        means = self._poisson_means()

        counts_a = self.rng.poisson(lam=means["team_a"])
        counts_b = self.rng.poisson(lam=means["team_b"])

        event_times = []
        delta_a = []
        delta_b = []

        for index, count in enumerate(counts_a, start=1):
            if count > 0:
                times = self.rng.uniform(0.0, horizon, size=int(count))
                event_times.extend(times.tolist())
                delta_a.extend([index] * int(count))
                delta_b.extend([0] * int(count))

        for index, count in enumerate(counts_b, start=1):
            if count > 0:
                times = self.rng.uniform(0.0, horizon, size=int(count))
                event_times.extend(times.tolist())
                delta_a.extend([0] * int(count))
                delta_b.extend([index] * int(count))

        if event_times:
            order = np.argsort(np.asarray(event_times))
            event_times = np.asarray(event_times, dtype=float)[order]
            delta_a = np.asarray(delta_a, dtype=int)[order]
            delta_b = np.asarray(delta_b, dtype=int)[order]
        else:
            event_times = np.array([], dtype=float)
            delta_a = np.array([], dtype=int)
            delta_b = np.array([], dtype=int)

        score_a = np.concatenate(([0], np.cumsum(delta_a)))
        score_b = np.concatenate(([0], np.cumsum(delta_b)))
        times = np.concatenate(([0.0], event_times))

        if times[-1] < horizon:
            times = np.append(times, horizon)
            score_a = np.append(score_a, score_a[-1])
            score_b = np.append(score_b, score_b[-1])

        margin = score_a - score_b

        return {
            "times": times,
            "score_a": score_a,
            "score_b": score_b,
            "margin": margin,
            "final_score_a": int(score_a[-1]),
            "final_score_b": int(score_b[-1]),
        }

    def simulate(self):
        weights = np.array([1, 2, 3], dtype=int)
        means = self._poisson_means()

        team_a_counts = self.rng.poisson(lam=means["team_a"], size=(self.num_simulations, 3))
        team_b_counts = self.rng.poisson(lam=means["team_b"], size=(self.num_simulations, 3))
        margins = team_a_counts @ weights - team_b_counts @ weights

        self.margins = margins.astype(int)
        return self.margins.copy()

    def simulate_paths(self, num_paths=5):
        self.paths = [self._single_path() for _ in range(int(num_paths))]
        return self.paths

    def price(self):
        margins = self.simulate()
        payoffs = self._payoff(margins)
        estimate = float(payoffs.mean())
        standard_error, ci = self._confidence_interval(estimate, len(payoffs))

        self.payoffs = payoffs
        self.price_estimate = estimate
        self.std_error = standard_error
        self.ci = ci

        return {
            "price": estimate,
            "standard_error": standard_error,
            "confidence_interval": ci,
            "num_simulations": self.num_simulations,
            "seed": self.seed,
        }

    def convergence(self, checkpoints: Iterable[int]):
        if self.payoffs is None:
            self.price()

        checkpoints = np.array(sorted(set(int(n) for n in checkpoints)), dtype=int)

        cumulative_payoffs = np.cumsum(self.payoffs)
        estimates = cumulative_payoffs[checkpoints - 1] / checkpoints
        standard_errors = np.sqrt(estimates * (1.0 - estimates) / checkpoints)
        ci_low = np.maximum(0.0, estimates - 1.96 * standard_errors)
        ci_high = np.minimum(1.0, estimates + 1.96 * standard_errors)

        return {
            "checkpoints": checkpoints,
            "estimates": estimates,
            "standard_errors": standard_errors,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }


def price_from_state(current_margin, time_remaining, lambda_a, lambda_b, p, num_simulations, seed=42):
    """
    Price the Team A winner contract from an in-game state.

    The remaining game is simulated over ``time_remaining`` using the supplied
    intensities and pooled mark distribution. The current margin is then added
    to each simulated future margin.
    """

    current_margin = int(current_margin)
    time_remaining = float(time_remaining)

    if time_remaining <= 0.0:
        price = float(current_margin > 0)
        standard_error = 0.0
        confidence_interval = (price, price)
        return {
            "price": price,
            "standard_error": standard_error,
            "confidence_interval": confidence_interval,
            "num_simulations": int(num_simulations),
            "seed": seed,
        }

    contract = {
        "lambda_a": float(lambda_a),
        "lambda_b": float(lambda_b),
        "p_a": np.asarray(p, dtype=float),
        "p_b": np.asarray(p, dtype=float),
        "T": time_remaining,
        "contract_type": "team_a_yes",
    }

    pricer = MonteCarlo(contract, num_simulations=num_simulations, seed=seed)
    future_margins = pricer.simulate()
    final_margins = current_margin + future_margins
    payoffs = (final_margins > 0).astype(float)

    estimate = float(payoffs.mean())
    standard_error, ci = pricer._confidence_interval(estimate, len(payoffs))

    return {
        "price": estimate,
        "standard_error": standard_error,
        "confidence_interval": ci,
        "num_simulations": int(num_simulations),
        "seed": seed,
    }


def update_total_pace(lambda_total_0, elapsed_minutes, total_scoring_events_observed, prior_weight_minutes=12.0):
    """
    Update the total scoring-event intensity using a Gamma-Poisson style shrinkage rule.
    """

    elapsed_minutes = float(elapsed_minutes)
    prior_weight_minutes = float(prior_weight_minutes)

    return (
        prior_weight_minutes * float(lambda_total_0) + float(total_scoring_events_observed)
    ) / (prior_weight_minutes + elapsed_minutes)
