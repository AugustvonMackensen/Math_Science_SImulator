"""Demo: stochastic calculus — GBM paths, Ito integral, Monte-Carlo pricing.

Run with:  .venv\\Scripts\\python demo_sde.py
"""

from __future__ import annotations

import numpy as np

from maths.stochastic import (
    geometric_brownian_motion,
    ito_integral,
    monte_carlo_expectation,
    ornstein_uhlenbeck,
)


def main() -> None:
    # 1) Geometric Brownian motion: E[X_T] = x0 * exp(mu T).
    x0, mu, sigma, T = 1.0, 0.08, 0.25, 1.0
    gbm = geometric_brownian_motion(x0, mu, sigma, (0.0, T),
                                    n_steps=500, n_paths=50000, seed=0)
    print("Geometric Brownian motion")
    print(f"   E[X_T] sampled = {gbm.terminal.mean():.4f}   "
          f"closed form = {x0 * np.exp(mu * T):.4f}")
    lo, hi = gbm.confidence_band(0.95)
    print(f"   95% band at T  = [{lo[-1]:.3f}, {hi[-1]:.3f}]")

    # 2) Monte-Carlo expectation of a call-style payoff E[max(X_T - K, 0)].
    est, se = monte_carlo_expectation(lambda xt: np.maximum(xt - 1.0, 0.0), gbm)
    print(f"   E[max(X_T - 1, 0)] = {est:.4f} +/- {se:.4f}")

    # 3) Ornstein-Uhlenbeck mean reversion toward mu.
    ou = ornstein_uhlenbeck(5.0, theta=1.5, mu=0.0, sigma=0.3, t_span=(0.0, 3.0),
                            n_steps=600, n_paths=20000, seed=1)
    print("\nOrnstein-Uhlenbeck")
    print(f"   E[X_T] sampled = {ou.terminal.mean():.4f}   (reverting to mu = 0)")

    # 4) Ito integral signature: E[(\int_0^T 1 dW)^2] = T  (Ito isometry).
    vals = ito_integral(lambda t, w: np.ones_like(w), (0.0, 2.0),
                        n_steps=2000, n_paths=40000, seed=2)
    print("\nIto isometry check")
    print(f"   E[(int 1 dW)^2] = {(vals**2).mean():.4f}   (should equal T = 2.0)")


if __name__ == "__main__":
    main()
