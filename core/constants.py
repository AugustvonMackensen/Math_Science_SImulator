"""Physical constants (CODATA 2018 recommended values).

Each constant is a :class:`PhysicalConstant` carrying its value, SI unit,
standard uncertainty, and symbol — so the engine can do dimensional
bookkeeping and report uncertainties, not just bare floats.

Examples
--------
>>> from core.constants import CONSTANTS, constant
>>> CONSTANTS["c"].value
299792458.0
>>> constant("hbar").value  # reduced Planck constant
1.0545718176461565e-34
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

from .exceptions import ModelError


@dataclass(frozen=True, slots=True)
class PhysicalConstant:
    """A physical constant with metadata.

    Parameters
    ----------
    symbol
        Conventional symbol, e.g. ``"c"`` or ``"hbar"``.
    name
        Human-readable name.
    value
        Numerical value in SI base units.
    unit
        SI unit string, e.g. ``"m s^-1"``.
    uncertainty
        Standard (1-sigma) uncertainty in the same unit. ``0.0`` means the
        constant is exact by definition (post-2019 SI redefinition).
    """

    symbol: str
    name: str
    value: float
    unit: str
    uncertainty: float = 0.0

    @property
    def is_exact(self) -> bool:
        """True if the constant is exact by SI definition."""
        return self.uncertainty == 0.0

    @property
    def relative_uncertainty(self) -> float:
        """Relative standard uncertainty (dimensionless); 0.0 if exact."""
        if self.value == 0.0:
            return 0.0
        return self.uncertainty / abs(self.value)

    def __float__(self) -> float:
        return self.value


def _build_registry() -> dict[str, PhysicalConstant]:
    # Exact constants (SI 2019 redefinition) carry uncertainty 0.0.
    c = 299792458.0  # speed of light, m s^-1 (exact)
    h = 6.62607015e-34  # Planck constant, J s (exact)
    e = 1.602176634e-19  # elementary charge, C (exact)
    k_B = 1.380649e-23  # Boltzmann constant, J K^-1 (exact)
    N_A = 6.02214076e23  # Avogadro constant, mol^-1 (exact)

    raw: list[PhysicalConstant] = [
        PhysicalConstant("c", "speed of light in vacuum", c, "m s^-1", 0.0),
        PhysicalConstant("h", "Planck constant", h, "J s", 0.0),
        PhysicalConstant("hbar", "reduced Planck constant", h / (2 * pi), "J s", 0.0),
        PhysicalConstant("e", "elementary charge", e, "C", 0.0),
        PhysicalConstant("k_B", "Boltzmann constant", k_B, "J K^-1", 0.0),
        PhysicalConstant("N_A", "Avogadro constant", N_A, "mol^-1", 0.0),
        PhysicalConstant("G", "Newtonian gravitation", 6.67430e-11, "m^3 kg^-1 s^-2", 1.5e-15),
        PhysicalConstant("g", "standard gravity", 9.80665, "m s^-2", 0.0),
        PhysicalConstant("m_e", "electron mass", 9.1093837015e-31, "kg", 2.8e-40),
        PhysicalConstant("m_p", "proton mass", 1.67262192369e-27, "kg", 5.1e-37),
        PhysicalConstant("m_n", "neutron mass", 1.67492749804e-27, "kg", 9.5e-37),
        PhysicalConstant("alpha", "fine-structure constant", 7.2973525693e-3, "1", 1.1e-12),
        PhysicalConstant("R", "molar gas constant", 8.314462618, "J mol^-1 K^-1", 0.0),
        PhysicalConstant("eps_0", "vacuum permittivity", 8.8541878128e-12, "F m^-1", 1.3e-21),
        PhysicalConstant("mu_0", "vacuum permeability", 1.25663706212e-6, "N A^-2", 1.9e-16),
        PhysicalConstant("sigma_SB", "Stefan-Boltzmann constant", 5.670374419e-8, "W m^-2 K^-4", 0.0),
        PhysicalConstant("a_0", "Bohr radius", 5.29177210903e-11, "m", 8.0e-21),
        PhysicalConstant("R_inf", "Rydberg constant", 1.0973731568160e7, "m^-1", 2.1e-5),
    ]
    return {pc.symbol: pc for pc in raw}


#: Immutable registry of physical constants keyed by symbol.
CONSTANTS: dict[str, PhysicalConstant] = _build_registry()


def constant(symbol: str) -> PhysicalConstant:
    """Look up a physical constant by symbol.

    Raises
    ------
    ModelError
        If the symbol is not registered.
    """
    try:
        return CONSTANTS[symbol]
    except KeyError as exc:
        known = ", ".join(sorted(CONSTANTS))
        raise ModelError(f"unknown constant {symbol!r}; known: {known}") from exc
