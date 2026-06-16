# Math & Science Simulator

A postgraduate-level compute engine for **physics** and **mathematics**, built to power a
web app with an in-browser **Code IDE** and **terminal** output.

> **Status:** Full stack in place — compute engine (Python), FastAPI backend with a
> Dockerized execution sandbox, and a React + TypeScript frontend (Monaco IDE,
> KaTeX formulas, geometry canvas). Monorepo.

## Architecture

```
frontend/  React + TypeScript (Monaco IDE · KaTeX formulas · geometry SVG)
        │  HTTP (Vite proxy /api, /health)
backend/   FastAPI  ──► Docker sandbox (per-run, --network none, resource-limited)
        │              └► dev-only local executor fallback
        │
   compute engine  (core / maths / physics / stats)  ◄── repo root
```

## Running the web app

```bash
# 1) engine + backend deps (one time)
.venv\Scripts\python -m pip install -e ".[dev]" -r backend/requirements.txt

# 2) backend on :8000
.venv\Scripts\python -m uvicorn backend.app.main:app --reload

# 3) frontend on :5173 (new terminal)
cd frontend && npm install && npm run dev
```

See [backend/README.md](backend/README.md) and [frontend/README.md](frontend/README.md) for details.

## Packages

| Package    | Contents                                                                 |
| ---------- | ------------------------------------------------------------------------ |
| `core`               | Physical constants (CODATA 2018, with units & uncertainties), exceptions |
| `maths.ode`          | Numerical ODE integration: adaptive (`integrate`) + symplectic (`velocity_verlet`) |
| `maths.stochastic`   | Stochastic calculus: Brownian motion, Itô SDEs (Euler-Maruyama, Milstein), Itô integrals, GBM/Ornstein-Uhlenbeck, Monte-Carlo |
| `maths.linalg`       | Decompositions (LU/QR/Cholesky/SVD/eigen), Gram-Schmidt, null space, conditioning, matrix exponential |
| `maths.calculus`     | Symbolic differentiation, integration, limits, Taylor series; vector calculus (grad/div/curl/Laplacian), Jacobian/Hessian |
| `maths.pde`          | 1-D PDEs: heat (Crank-Nicolson) and wave (leapfrog) with CFL checks |
| `maths.geometry`     | **Euclidean/plane** (lines, circles, triangles, intersections, convex hull), affine **transforms**, **non-Euclidean** (spherical + hyperbolic) |
| `stats`              | Distributions (uniform API), estimation (MLE, CIs, bootstrap), hypothesis tests (t/ANOVA/χ²/KS/Shapiro/correlation) |
| `physics.mechanics`  | Symbolic **Lagrangian** and **Hamiltonian** engines |
| `physics.quantum`    | 1-D time-independent **Schrödinger** eigensolver (finite differences) |
| `physics.electromagnetism` | Point-charge fields/potentials (Coulomb superposition) + **Laplace** boundary-value solver |
| `physics.statistical`| Canonical ensemble (partition function, energy, entropy, heat capacity) + Maxwell-Boltzmann ideal gas |

## Quick start

```bash
py -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
```

## Example — simple pendulum from its Lagrangian

The user types a Lagrangian; the engine derives the Euler–Lagrange equations
symbolically, solves for the accelerations, and integrates the motion.

```python
from physics.mechanics import LagrangianSystem

pendulum = LagrangianSystem(
    coordinates=["theta"],
    parameters=["m", "l", "g"],
    lagrangian="m*l**2*theta_dot**2/2 + m*g*l*cos(theta)",
)

# Symbolic equation of motion:  θ̈ = -(g/l) sin θ
print(pendulum.acceleration_expressions()["theta_ddot"])

result = pendulum.simulate(
    initial={"theta": (1.0, 0.0)},          # 1 rad, released from rest
    t_span=(0.0, 15.0),
    parameters={"m": 1.0, "l": 1.0, "g": 9.81},
)
theta = result.component(0)                 # angle vs. result.t
```

Run the bundled demo:

```bash
.venv\Scripts\python demo.py
```

## Design notes

- **Symbolic + numeric.** SymPy derives the exact equations of motion; SciPy/`numpy`
  integrate them after `lambdify`. This separation is what makes arbitrary
  user-supplied Lagrangians possible.
- **Energy as a correctness check.** Each `LagrangianSystem` exposes its conserved
  Hamiltonian; tests assert energy drift `< 1e-6` over the trajectory.
- **Symplectic option.** `velocity_verlet` gives bounded long-horizon energy error
  for conservative systems, unlike plain Runge–Kutta.

## Roadmap

- [x] Stochastic calculus: Itô SDEs (Euler-Maruyama, Milstein), Itô integrals, GBM/OU
- [x] Linear algebra, symbolic + vector calculus, 1-D PDE solvers
- [x] Geometry: Euclidean/plane, affine transforms, non-Euclidean (spherical + hyperbolic)
- [x] Statistics: distributions, estimators (MLE/CI/bootstrap), hypothesis tests (`stats/` package — separate from stdlib `statistics`)
- [x] Hamiltonian mechanics; quantum (time-independent Schrödinger); electromagnetism; statistical mechanics
- [x] FastAPI backend (`backend/`): code execution, formula rendering/calculus, geometry scenes — see [backend/README.md](backend/README.md)
- [x] Docker execution sandbox (`--network none`, resource limits) + dev local fallback
- [x] React + TS frontend (`frontend/`): Monaco IDE + terminal/figures, KaTeX formula panel, geometry SVG canvas — see [frontend/README.md](frontend/README.md)
- [ ] Deeper physics: rigid bodies, constraints (Lagrange multipliers), time-dependent Schrödinger, magnetostatics (Biot-Savart)
- [ ] More maths: 2-D/3-D PDEs, optimization, number theory, abstract algebra
- [ ] Frontend polish: persistent sessions, multiple files, xterm.js interactive terminal, plot interactivity
```
