from .unit_conversion import extract_context, Q_, quantity_factory

__all__ = ["Fluid"]

class Fluid:
    """
    A class to represent a fluid and its properties, enabling engineering calculations.
    
    Attributes
    ----------
    name : str
        The name of the fluid.
    density : Q_
        The density of the fluid in standard units.
    """

    def __init__(self, name: str, density: Q_, **kwargs) -> None:
        """
        Initializes a Fluid object with converted physical properties.

        Parameters
        ----------
        name : str
            The name of the fluid.
        density : Q_
            The density of the fluid with units (e.g., `Q_(1000, "kg/m**3")`).
        kwargs : dict
            Additional physical properties with units.

        Raises
        ------
        ValueError
            If the provided units are not compatible.
        """
        if not name or not isinstance(name, str):
            raise ValueError("A valid name for the fluid is required.")
        if not density:
            raise ValueError("Density is a mandatory property.")

        self.name = name
        self.density = quantity_factory(density, context="default")
        
        for key, quantity in kwargs.items():
            context = extract_context(key)
            setattr(self, key, quantity_factory(quantity, context))

    def __repr__(self) -> str:
        """Returns a detailed string representation of the fluid."""
        return f"Fluid(name={self.name}, density={self.density})"
    
    def __str__(self) -> str:
        """Returns a detailed string representation of the fluid's properties."""
        properties = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"Fluid({properties})"
