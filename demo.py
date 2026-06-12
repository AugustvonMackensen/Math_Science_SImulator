"""Demo: derive and simulate a simple pendulum from its Lagrangian.

Run with:  .venv\\Scripts\\python demo.py
"""

from __future__ import annotations

import numpy as np

from physics.mechanics import LagrangianSystem


def main() -> None:
    pendulum = LagrangianSystem(
        coordinates=["theta"],
        parameters=["m", "l", "g"],
        lagrangian="m*l**2*theta_dot**2/2 + m*g*l*cos(theta)",
    )

    print("Euler-Lagrange equation:")
    for eq in pendulum.euler_lagrange_equations():
        print("   ", eq)
    print("Acceleration:  theta_ddot =", pendulum.acceleration_expressions()["theta_ddot"])
    print("Energy (Hamiltonian):", pendulum.energy_expression())

    params = {"m": 1.0, "l": 1.0, "g": 9.81}
    result = pendulum.simulate(
        initial={"theta": (1.0, 0.0)},
        t_span=(0.0, 15.0),
        parameters=params,
        n_points=1500,
    )

    energy = pendulum.energy(result.y, params)
    drift = np.abs(energy - energy[0]).max() / abs(energy[0])

    print(f"\nIntegrated {result.t.size} samples with {result.method}.")
    print(f"Max |theta| = {np.abs(result.component(0)).max():.4f} rad")
    print(f"Relative energy drift = {drift:.2e}")


if __name__ == "__main__":
    main()
