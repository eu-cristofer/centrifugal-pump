"""
utilities
=========

This submodule provides unit conversion utilities to standardize physical quantities
using Pint.

Overview
--------
The `utilities` module serves as a central point for handling unit conversions, working with
fluid properties, and standardizing physical quantities using the Pint library.

Classes
-------
- UnitConverterInterface: Defines an interface for unit conversion.
- ImprovedQuantity: Implements unit conversion logic.
- Fluid: Represents a fluid and its physical properties.

Functions
---------
- quantity_factory: A helper function for unit conversion.

Constants
---------
- CONTEXT: Set of available conversion contexts.
- STANDARD_UNITS: Dictionary defining standard units for various physical quantities.

Examples
--------
Basic fluid representation and conversion:

>>> from utilities import Fluid, Q_
>>> water = Fluid(name="Water", density=Q_(1000, "kg/m**3"))
>>> print(water)
Fluid(name=Water, density=1.0 kilogram / meter ** 3)

>>> from utilities import quantity_factory, Q_
>>> flow_rate = quantity_factory(Q_(10, "liter/minute"))
>>> print(flow_rate)
0.00016666666666666666 meter ** 3 / second
"""
from . import unit_conversion, fluid, report
from .unit_conversion import *
from .fluid import *
from .report import *

__all__ = list(
    set(unit_conversion.__all__) |
    set(fluid.__all__) |
    set(report.__all__)
)