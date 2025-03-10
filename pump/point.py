"""
point
=====

This module defines the Point class and its subclasses, which represent a system point 
with converted physical quantities.

Classes
-------
- Point: Represents a system point blueprint with capacity and inlet pressure.
- DesignPoint: A specialized version of Point for design conditions.
- TestPoint: A specialized version of Point for handling test or operational data.
"""

import numpy as np
from .utilities.unit_conversion import extract_context, Q_, quantity_factory
from .utilities.fluid import Fluid

__all__ = ["Point", "DesignPoint", "TestPoint"]

class Point:
    """
    Represents a system point with physical quantities.

    Parameters
    ----------
    fluid : Fluid
        A fluid object that should at least contain properties like density.
    capacity : Q_
        The capacity as a Pint quantity with units (e.g., Q_(10, "m**3/s")).
    **kwargs : dict[str, Q_]
        Additional named quantities. 

    Attributes
    ----------
    capacity : Q_
        The capacity converted to its standard unit.
    fluid : Fluid
        The fluid associated with the point.
    <other attributes from kwargs> : Q_
        Any additional attributes passed via kwargs are stored as Pint
        quantities in this object.

    Raises
    ------
    ValueError
        If the provided units are not compatible or if quantity_factory
        raises an error for invalid unit conversion.

    Examples
    --------
    >>> from pump.utilities.unit_conversion import Fluid, Q_
    >>> water = Fluid(name="Water", density=Q_(1000, "kg/m**3"))
    >>> point = Point(fluid=water, capacity=Q_(0.1, "m**3/s"), inlet_pressure=Q_(1, "atm"))
    >>> print(point)
    Point(fluid=Water, capacity=0.1 meter ** 3 / second, inlet_pressure=101325.0 pascal)
    """

    def __init__(self, fluid: Fluid, capacity: Q_, **kwargs: dict[str, Q_]) -> None:
        self.fluid: Fluid = fluid
        self.capacity: Q_ = quantity_factory(capacity, context="default")
        for key, quantity in kwargs.items():
            context = extract_context(key)
            setattr(self, key, quantity_factory(quantity, context))
        
    def __repr__(self) -> str:
        return f"Point(name={self.fluid.name}, capacity={self.capacity:.2f~P})"

    def __str__(self) -> str:
        properties = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"Point({properties})"


class DesignPoint(Point):
    """
    Specialized version of the Point class, representing a design condition.
    """
    def __init__(self, fluid: Fluid, capacity: Q_, head: Q_, **kwargs: dict[str, Q_]) -> None:
        kwargs["head"] = head
        super().__init__(fluid, capacity, **kwargs)


class TestPoint(Point):
    """
    Specialized version of the Point class for handling test or operational data.

    This class provides additional properties for calculating hydraulic parameters
    such as pressure head, velocity head, elevation head, and hydraulic power.
    """

    g = Q_(9.81, "m/s**2")

    @property
    def pressure_head(self) -> Q_:
        if not hasattr(self.fluid, "density"):
            raise ValueError("Fluid object does not have a 'density' attribute.")

        if not hasattr(self, "delta_pressure"):
            if not hasattr(self, "inlet_pressure") or not hasattr(self, "outlet_pressure"):
                raise AttributeError("Cannot compute head because 'delta_pressure' or pressures are missing.")
            else:
                self.delta_pressure = self.outlet_pressure - self.inlet_pressure
        return quantity_factory(self.delta_pressure / (self.fluid.density * self.g))

    @property
    def inlet_velocity(self) -> Q_:
        if hasattr(self, "inlet_diameter"):
            return quantity_factory(self.capacity / (np.pi * self.inlet_diameter**2 / 4))
        return Q_(0, "m/s")

    @property
    def outlet_velocity(self) -> Q_:
        if hasattr(self, "outlet_diameter"):
            return quantity_factory(self.capacity / (np.pi * self.outlet_diameter**2 / 4))
        return Q_(0, "m/s")

    @property
    def velocity_head(self) -> Q_:
        if hasattr(self, "inlet_diameter") and hasattr(self, "outlet_diameter"):
            return quantity_factory((self.outlet_velocity**2 - self.inlet_velocity**2) / (2 * self.g))
        return Q_(0, "m")

    @property
    def elevation_head(self) -> Q_:
        if hasattr(self, "inlet_elevation") and hasattr(self, "outlet_elevation"):
            return quantity_factory(self.outlet_elevation - self.inlet_elevation)
        return Q_(0, "m")

    @property
    def compute_head(self) -> Q_:
        TDH = self.pressure_head + self.velocity_head + self.elevation_head
        self._head = quantity_factory(TDH)
        return self._head

    @property
    def head(self) -> Q_:
        return self._head if hasattr(self, "_head") else self.compute_head

    @property
    def compute_hydraulic_power(self) -> Q_:
        self._hydraulic_power = quantity_factory(self.fluid.density * self.capacity * self.g * self.head)
        return self._hydraulic_power

    @property
    def hydraulic_power(self) -> Q_:
        return self._hydraulic_power if hasattr(self, "_hydraulic_power") else self.compute_hydraulic_power

    @property
    def compute_efficiency(self) -> Q_:
        if not hasattr(self, "breaking_power"):
            raise AttributeError("Cannot compute efficiency: 'breaking_power' is missing.")
        self._efficiency = quantity_factory(self.hydraulic_power / self.breaking_power).to("percent")
        return self._efficiency

    @property
    def efficiency(self) -> Q_:
        return self._efficiency if hasattr(self, "_efficiency") else self.compute_efficiency

    def __lt__(self, other: "TestPoint") -> bool:
        return self.capacity < other.capacity
