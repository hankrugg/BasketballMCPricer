"""
Monte Carlo pricer for basketball winner contracts under the baseline model.
"""

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

    def _poisson_means(self):
        """
        Convert scoring intensities into Poisson means for 1-, 2-, and 3-point event counts.

        For team i and mark k, N_{i,k}(T) has mean lambda_i * p_{i,k} * T.
        """
        horizon = float(self.contract["T"])
        p_a = np.asarray(self.contract["p_a"], dtype=float)
        p_b = np.asarray(self.contract["p_b"], dtype=float)
        return {
            "team_a": float(self.contract["lambda_a"]) * p_a * horizon,
            "team_b": float(self.contract["lambda_b"]) * p_b * horizon,
        }

    def _confidence_interval(self, estimate, n_paths):
        standard_error = np.sqrt(estimate * (1.0 - estimate) / n_paths)
        ci_low = max(0.0, estimate - 1.96 * standard_error)
        ci_high = min(1.0, estimate + 1.96 * standard_error)
        return standard_error, (ci_low, ci_high)

    def simulate(self):
        """
        Simulate terminal point margins for many independent games.

        This is the fast pricing path: the winner payoff depends only on the
        final margin, so there is no need to simulate event times.
        """
        weights = np.array([1, 2, 3], dtype=int)
        means = self._poisson_means()

        # Each row is one simulated game; columns are 1-, 2-, and 3-point event counts.
        team_a_counts = self.rng.poisson(lam=means["team_a"], size=(self.num_simulations, 3))
        team_b_counts = self.rng.poisson(lam=means["team_b"], size=(self.num_simulations, 3))

        # Convert event counts into scores and subtract to get Team A's final margin.
        margins = team_a_counts @ weights - team_b_counts @ weights
        return margins.astype(int)

    def price(self):
        """
        Estimate the Team A yes price as the average simulated winner payoff.
        """
        margins = self.simulate()

        # The digital payoff is 1 when Team A's final margin is positive.
        payoffs = (margins > 0).astype(float)
        estimate = float(payoffs.mean())
        standard_error, ci = self._confidence_interval(estimate, len(payoffs))

        return {
            "price": estimate,
            "standard_error": standard_error,
            "confidence_interval": ci,
            "num_simulations": self.num_simulations,
            "seed": self.seed,
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

    # Once regulation time is exhausted, the payoff is already determined.
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

    # Reuse the pregame MonteCarlo class on the remaining horizon only.
    # The same pooled mark distribution is used for both teams in this notebook.
    contract = {
        "lambda_a": float(lambda_a),
        "lambda_b": float(lambda_b),
        "p_a": np.asarray(p, dtype=float),
        "p_b": np.asarray(p, dtype=float),
        "T": time_remaining,
    }

    pricer = MonteCarlo(contract, num_simulations=num_simulations, seed=seed)
    future_margins = pricer.simulate()

    # Current margin is known; simulation only supplies the future margin change.
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
