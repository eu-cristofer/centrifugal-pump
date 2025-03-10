"""
unit_conversion
===============

This submodule provides utilities for handling unit conversions using the Pint library.

Overview
--------
The submodule defines an interface and an implementation for unit conversion, ensuring that 
quantities are converted into predefined standard units.

Classes
-------
- UnitConverterInterface: Abstract base class defining a unit converter interface.
- ImprovedQuantity: Implements unit conversion functionality.

Functions
---------
- quantity_factory: A utility function to convert a given quantity to its standard unit.

Constants
---------
- STANDARD_UNITS: A dictionary defining the standard units for various physical quantities.

Dependencies
------------
- pint: A library for unit handling in Python.

Examples
--------
Basic unit conversion:

>>> from unit_conversion import quantity_factory, Q_
>>> mass = quantity_factory(Q_(500, "gram"))
>>> print(mass)
0.5 kilogram

>>> pressure = quantity_factory(Q_(1, "atm"), context="atm")
>>> print(pressure)
101325.0 pascal

>>> delta_pressure = quantity_factory(Q_(1, "atm"), context="delta")
>>> print(delta_pressure)
1.01325 bar
"""

from logging import getLogger, DEBUG
import warnings
from abc import ABC, abstractmethod
from typing import Dict
from pint import UnitRegistry, Quantity

__all__ = [
    "CONTEXT",
    "Q_",
    "quantity_factory",
    "STANDARD_UNITS"
]

# Initialize logging
logger = getLogger(__name__)
logger.setLevel(DEBUG)

# Initialize Pint's unit registry
ureg = UnitRegistry()
ureg.formatter.default_format = '~P'  # Abbreviated unit names.
Q_ = Quantity

# Standard units for various physical quantities
STANDARD_UNITS: Dict[str, Dict[str, str]] = {
    "capacity": {"default": "m**3/h"},
    "density": {"default": "kg/m**3"},
    "dynamic_viscosity": {"default": "cP"},
    "efficiency": {"default": "%"},
    "energy": {"default": "joule"},
    "length": {"default": "meter"},
    "mass": {"default": "kilogram"},
    "power": {"default": "kW"},
    "pressure": {"default": "bar", "atm": "pascal", "delta": "bar"},
    "speed_of_rotation": {"default": "rpm"},
    "velocity": {"default": "m/s"},
    "temperature": {"default": "kelvin", "delta": "kelvin"},
    "time": {"default": "second"},
}

# Extract context options
CONTEXT: set[str] = set()
for unit_dict in STANDARD_UNITS.values():
    CONTEXT.update(unit_dict.keys())

def extract_context(key: str) -> str:
    """
    Extracts the context from the property name.
    
    Parameters
    ----------
    key : str
        The property name.

    Returns
    -------
    str
        The context if found, otherwise "default".
    """
    parts = key.split("_")
    return parts[0] if len(parts) >= 2 and parts[0] in CONTEXT else "default"

class UnitConverterInterface(ABC):
    """
    Defines an abstract interface for unit conversion.
    
    Methods
    -------
    convert(quantity: Q_, context: str = "default") -> Q_
        Converts a given quantity to a standardized unit.
    """
    @abstractmethod
    def convert(self, quantity: Q_, context: str = "default") -> Q_:
        pass

class ImprovedQuantity(UnitConverterInterface):
    """
    Implements unit conversion based on predefined standard units.
    """
    @classmethod
    def convert(cls, quantity: Q_, context: str = "default") -> Q_:
        """
        Converts a given quantity to its corresponding standard unit.

        Parameters
        ----------
        quantity : Q_
            The quantity to be converted.
        context : str, optional
            The conversion context, which determines the target unit (default is "default").

        Returns
        -------
        Q_
            The converted quantity in the standard unit.

        Raises
        ------
        ValueError
            If the unit is invalid or conversion is not possible.
        """
        if context not in CONTEXT:
            raise ValueError(f"Invalid context: {context}. Allowed contexts: {CONTEXT}")
        
        try:
            dim = quantity.dimensionality
            for category, unit_map in STANDARD_UNITS.items():
                default_unit_dim = ureg(unit_map["default"]).dimensionality
                if dim == default_unit_dim:
                    standard_unit = unit_map.get(context, unit_map["default"])
                    
                    if category == "temperature" and context == "delta":
                        if "delta_" in str(quantity.units):
                            return quantity.to(standard_unit)
                        else:
                            return Q_(quantity.m, "delta_" + str(quantity.units)).to(standard_unit)
                    else:
                        return quantity.to(standard_unit)
        except pint.UndefinedUnitError:
            raise ValueError(f"Invalid unit: {quantity.units}")
        except pint.DimensionalityError as e:
            raise ValueError(f"Incompatible unit conversion: {e}")
        
        logger.debug(
            f"No standard unit match found for dimensionality {quantity.dimensionality}. "
            f"Returning quantity unchanged: {quantity}"
        )
        warnings.warn(
            f"No matching dimensionality found in STANDARD_UNITS for {quantity.units}. "
            "Returning the quantity unchanged.",
            UserWarning
        )
        return quantity

def quantity_factory(quantity: Q_, context: str = "default") -> Q_:
    """
    Converts a given quantity to its standard unit.
    
    Parameters
    ----------
    quantity : Q_
        The quantity to be converted.
    context : str, optional
        The conversion context (default is "default").

    Returns
    -------
    Q_
        The converted quantity in the standard unit.
    """
    return ImprovedQuantity.convert(quantity, context)

if __name__ == "__main__":
    mass = quantity_factory(Q_(500, "gram"))
    print(mass)
    
    atm_pressure = quantity_factory(Q_(1, "atm"), context="atm")
    print(atm_pressure)
    
    delta_pressure = quantity_factory(Q_(1, "atm"), context="delta")
    print(delta_pressure)
    
    temp_diff = quantity_factory(Q_(1, "degC"), context="delta")
    print(temp_diff)
