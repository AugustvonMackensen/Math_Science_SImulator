r"""Statistical mechanics — the canonical ensemble and the ideal gas.

From a discrete energy spectrum this module builds the partition function,
Boltzmann occupation probabilities, and the derived thermodynamics (internal
energy, Helmholtz free energy, Gibbs entropy, heat capacity). It also provides
the Maxwell-Boltzmann speed distribution and its characteristic speeds for an
ideal gas.

Temperatures are in kelvin; ``k_B`` is taken from :mod:`core.constants`. Pass
``k_B=1.0`` explicitly to work in natural (energy = temperature) units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.constants import constant
from core.exceptions import ModelError

_KB = constant("k_B").value


def _beta(temperature: float, k_B: float) -> float:
    if temperature <= 0:
        raise ModelError("temperature must be positive")
    return 1.0 / (k_B * temperature)


def partition_function(
    energies, temperature: float, *, degeneracies=None, k_B: float = _KB
) -> float:
    r"""Canonical partition function :math:`Z = \sum_i g_i e^{-\beta E_i}`.

    A constant is subtracted from the exponent for numerical stability (it
    cancels in all probabilities/averages but not in ``Z`` itself, so ``Z``
    here is relative to the ground state).
    """
    E = np.asarray(energies, dtype=float)
    g = np.ones_like(E) if degeneracies is None else np.asarray(degeneracies, dtype=float)
    beta = _beta(temperature, k_B)
    shifted = -beta * (E - E.min())
    return float(np.sum(g * np.exp(shifted)))


def boltzmann_probabilities(
    energies, temperature: float, *, degeneracies=None, k_B: float = _KB
) -> np.ndarray:
    r"""Occupation probabilities :math:`p_i = g_i e^{-\beta E_i}/Z`."""
    E = np.asarray(energies, dtype=float)
    g = np.ones_like(E) if degeneracies is None else np.asarray(degeneracies, dtype=float)
    beta = _beta(temperature, k_B)
    w = g * np.exp(-beta * (E - E.min()))
    return w / w.sum()


def internal_energy(
    energies, temperature: float, *, degeneracies=None, k_B: float = _KB
) -> float:
    r"""Ensemble-averaged energy :math:`\langle E\rangle = \sum_i p_i E_i`."""
    E = np.asarray(energies, dtype=float)
    p = boltzmann_probabilities(E, temperature, degeneracies=degeneracies, k_B=k_B)
    return float(np.sum(p * E))


def free_energy(
    energies, temperature: float, *, degeneracies=None, k_B: float = _KB
) -> float:
    r"""Helmholtz free energy :math:`F = -k_B T \ln Z` (relative to the ground state)."""
    Z = partition_function(energies, temperature, degeneracies=degeneracies, k_B=k_B)
    return float(-k_B * temperature * np.log(Z))


def entropy(
    energies, temperature: float, *, degeneracies=None, k_B: float = _KB
) -> float:
    r"""Gibbs entropy :math:`S = -k_B \sum_i p_i \ln p_i`."""
    p = boltzmann_probabilities(energies, temperature, degeneracies=degeneracies, k_B=k_B)
    nz = p[p > 0]
    return float(-k_B * np.sum(nz * np.log(nz)))


def heat_capacity(
    energies, temperature: float, *, degeneracies=None, k_B: float = _KB
) -> float:
    r"""Heat capacity :math:`C = (\langle E^2\rangle-\langle E\rangle^2)/(k_B T^2)`."""
    E = np.asarray(energies, dtype=float)
    p = boltzmann_probabilities(E, temperature, degeneracies=degeneracies, k_B=k_B)
    mean = np.sum(p * E)
    mean_sq = np.sum(p * E**2)
    return float((mean_sq - mean**2) / (k_B * temperature**2))


@dataclass(slots=True)
class TwoLevelSystem:
    """A two-level system with gap ``epsilon`` — the canonical toy model."""

    epsilon: float
    k_B: float = _KB

    def heat_capacity(self, temperature: float) -> float:
        """Schottky heat capacity, peaking near ``k_B T ~ epsilon``."""
        x = self.epsilon / (self.k_B * temperature)
        if x > 700.0:  # exp would overflow; C -> 0 as the gap freezes out
            return 0.0
        # Numerically stable form: e^x/(1+e^x)^2 = 1/(4 cosh^2(x/2)).
        return float(self.k_B * x**2 / (4.0 * np.cosh(x / 2.0) ** 2))


# --- ideal gas: Maxwell-Boltzmann -----------------------------------------

def maxwell_boltzmann_speed_pdf(
    speed, temperature: float, mass: float, *, k_B: float = _KB
) -> np.ndarray:
    r"""Maxwell-Boltzmann speed distribution :math:`f(v)` (3-D)."""
    v = np.asarray(speed, dtype=float)
    a = mass / (2.0 * k_B * temperature)
    return 4.0 * np.pi * (a / np.pi) ** 1.5 * v**2 * np.exp(-a * v**2)


def rms_speed(temperature: float, mass: float, *, k_B: float = _KB) -> float:
    r""":math:`v_\mathrm{rms} = \sqrt{3 k_B T/m}`."""
    return float(np.sqrt(3.0 * k_B * temperature / mass))


def mean_speed(temperature: float, mass: float, *, k_B: float = _KB) -> float:
    r""":math:`\bar v = \sqrt{8 k_B T/(\pi m)}`."""
    return float(np.sqrt(8.0 * k_B * temperature / (np.pi * mass)))


def most_probable_speed(temperature: float, mass: float, *, k_B: float = _KB) -> float:
    r""":math:`v_p = \sqrt{2 k_B T/m}`."""
    return float(np.sqrt(2.0 * k_B * temperature / mass))
