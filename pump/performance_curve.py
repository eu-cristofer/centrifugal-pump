"""
peroformance_curve
==================

This module defines the PerformanceCurve class, which represents a system point with converted
physical quantities.

Classes
-------
- PerformanceCurve: Represents a system point with capacity and inlet pressure.
- PerformanceChecker: Check the performance of a system based on its design point and performance curve.
- PerformanceFitter: Handles polynomial regression for performance metrics.
"""

from typing import Optional, List, Dict, Self, Any
import numpy as np
import matplotlib.pyplot as plt
from .utilities.unit_conversion import Q_, quantity_factory
from .utilities.fluid import Fluid
from .point import DesignPoint, TestPoint
from tabulate import tabulate
import io

__all__ = ["PerformanceChecker", "PerformanceCurve"]

class PerformanceFitter:
    """
    Handles polynomial regression for performance metrics (head, efficiency, power),
    with lazy computation of polynomial coefficients.

    Attributes
    ----------
    points : list of TestPoint
        A list of TestPoint instances used for polynomial regression.

    Methods
    -------
    fit_head()
        Lazily fits a 4th-order polynomial for head vs. capacity.
    fit_efficiency()
        Lazily fits a 4th-order polynomial for efficiency vs. capacity.
    fit_power()
        Lazily fits a 4th-order polynomial for power vs. capacity.
    """

    def __init__(self, points) -> None:
        self.points = points
        self._capacities = None
        self._heads = None
        self._efficiencies = None
        self._powers = None
        self._head_coeffs = None
        self._efficiency_coeffs = None
        self._power_coeffs = None
    
    @property
    def capacities(self) -> np.ndarray:
        """
        Lazily computes and returns the capacity values as a NumPy array.

        Returns
        -------
        np.ndarray
            Array of capacity values.
        """
        if self._capacities is None:
            self._capacities = np.array([p.capacity.magnitude for p in self.points])
        return self._capacities

    @property
    def heads(self) -> np.ndarray:
        """
        Lazily computes and returns the capacity values as a NumPy array.

        Returns
        -------
        np.ndarray
            Array of capacity values.
        """
        if self._heads is None:
            self._heads = np.array([p.head.magnitude for p in self.points])
        return self._heads

    @property
    def efficiencies(self) -> np.ndarray:
        """
        Lazily computes and returns the capacity values as a NumPy array.

        Returns
        -------
        np.ndarray
            Array of capacity values.
        """
        if self._efficiencies is None:
            self._efficiencies = np.array([p.efficiency.magnitude for p in self.points])
        return self._efficiencies

    @property
    def powers(self) -> np.ndarray:
        """
        Lazily computes and returns the capacity values as a NumPy array.

        Returns
        -------
        np.ndarray
            Array of capacity values.
        """
        
        if self._powers is None:
            self._powers = np.array([p.breaking_power.magnitude for p in self.points])
        return self._powers

    @property
    def head_coeffs(self) -> np.ndarray:
        """
        Lazily computes the 4th-order polynomial regression coefficients for head vs. capacity.

        Returns
        -------
        np.ndarray
            Array of polynomial coefficients.
        """
        if self._head_coeffs is None:
            capacities = self.capacities
            heads = np.array([p.head.magnitude for p in self.points])
            self._head_coeffs = np.polyfit(capacities, heads, 4)
        return self._head_coeffs

    @property
    def efficiency_coeffs(self) -> np.ndarray:
        """
        Lazily computes the 4th-order polynomial regression coefficients for efficiency vs. capacity.

        Returns
        -------
        np.ndarray
            Array of polynomial coefficients.
        """
        if self._efficiency_coeffs is None:
            capacities = self.capacities
            efficiencies = np.array([p.efficiency.magnitude for p in self.points])
            self._efficiency_coeffs = np.polyfit(capacities, efficiencies, 4)
        return self._efficiency_coeffs

    @property
    def power_coeffs(self) -> np.ndarray:
        """
        Lazily computes the 4th-order polynomial regression coefficients for power vs. capacity.

        Returns
        -------
        np.ndarray
            Array of polynomial coefficients.
        """
        if self._power_coeffs is None:
            capacities = self.capacities
            powers = np.array([p.breaking_power.magnitude for p in self.points])
            self._power_coeffs = np.polyfit(capacities, powers, 4)
        return self._power_coeffs
    
    

class PerformanceCurve:
    """
    A container for handling a collection of TestPoint objects that share the same fluid.

    This class allows you to:
      - Scale the pump speed to a new speed (using simplified pump affinity laws).
      - Change the fluid of the entire collection.
      - Always return a new instance when these transformations are applied,
        preserving the original data.
      - Predict performance metrics like head, efficiency, and power based on polynomial fits.
      - Plot pump performance curves.

    Parameters
    ----------
    fluid : Fluid
        The fluid object shared by all TestPoints in this collection.
    points : list of TestPoint
        A list of TestPoint instances that share the same fluid.

    Attributes
    ----------
    fluid : Fluid
        The fluid used by all points in this collection.
    points : list of TestPoint
        The list of performance points stored in this collection.

    Examples
    --------
    >>> # Suppose we have a list of TestPoint objects called original_points,
    >>> # each using the same fluid "water".
    >>> curve = PumpPerformanceCurve(water, original_points)
    >>>
    >>> # Convert pump speed from 3500 rpm to 4000 rpm
    >>> new_curve = curve.to_speed(Q_(4000, "rpm"))
    >>>
    >>> # Change fluid from water to some other fluid (e.g., oil)
    >>> oil_curve = curve.to_fluid(oil)
    """

    def __init__(self, fluid: Fluid, points: List[TestPoint]) -> None:
        self.fluid = fluid
        for pt in points:
            if pt.fluid != fluid:
                raise ValueError("All TestPoints must have the same fluid.")
        self.points = sorted(points) # Sort by capacity
        self.fitter = PerformanceFitter(points)

    def predict_metric(self, capacity, coeffs, unit) -> Q_:
        capacity_value = capacity.to("m**3/h").magnitude
        metric_value = np.polyval(coeffs, capacity_value)
        return quantity_factory(Q_(metric_value, unit))

    def predict_head(self, capacity: Q_) -> Q_:
        """
        Predicts the head for a given flow (capacity) using the stored polynomial.

        Parameters
        ----------
        capacity : Q_
            The input flow in m³/h.

        Returns
        -------
        Q_
            The predicted head in meters.
        """
        coeffs = self.fitter.head_coeffs
        return self.predict_metric(capacity, coeffs, "m")

    def predict_efficiency(self, capacity: Q_) -> Q_:
        """
        Predicts the efficiency for a given flow (capacity) using the stored polynomial.

        Parameters
        ----------
        capacity : Q_
            The input flow in m³/h.

        Returns
        -------
        Q_
            The predicted efficiency in percent.
        """
        coeffs = self.fitter.efficiency_coeffs
        return self.predict_metric(capacity, coeffs, "%")

    def predict_breaking_power(self, capacity: Q_) -> Q_:
        """
        Predicts the power for a given flow (capacity) using the stored polynomial.

        Parameters
        ----------
        capacity : Q_
            The input flow in m³/h.

        Returns
        -------
        Q_
            The predicted power in kW.
        """
        coeffs = self.fitter.power_coeffs
        return self.predict_metric(capacity, coeffs, "kW")

    def predicted_data(self, capacity:Q_) -> Dict:
        efficiency = self.predict_efficiency(capacity)
        head = self.predict_head(capacity)
        shutoff_head = self.predict_head(Q_(0, "m**3/h"))

        return {
            "Head": head,
            "Efficiency": efficiency,
            "Head Shuttoff": shutoff_head
        }

    def plot_performance_curve(self, chart_title=None, capacity:Q_=None, return_io=False):
        """
        Plots the pump performance curve with three subplots:
        - Head vs. Capacity (takes up more height)
        - Power vs. Capacity
        - Efficiency vs. Capacity
        """
        capacities = self.fitter.capacities
        heads = self.fitter.heads

        has_power = all(hasattr(p, "breaking_power") for p in self.points)
        has_efficiency = all(hasattr(p, "efficiency") for p in self.points)

        # Generate smooth values for polynomial curves
        smooth_capacities = np.linspace(min(capacities), max(capacities), 100)
        head_curve = np.polyval(self.fitter.head_coeffs, smooth_capacities)

        # Compute fitted values for the given flow
        if capacity is not None:
            capacity_value = capacity.to("m**3/h").magnitude
            fitted_head = np.polyval(self.fitter.head_coeffs, capacity_value)
    
            if has_power and has_efficiency:
                fitted_power = np.polyval(self.fitter.power_coeffs, capacity_value)
                fitted_efficiency = np.polyval(self.fitter.efficiency_coeffs, capacity_value)

        # Setup figure based on available data
        if has_power and has_efficiency:
            fig, axes = plt.subplots(
                3, 1, sharex=True, figsize=(8, 8), gridspec_kw={'height_ratios': [5, 3, 4]}
            )
        else:
            fig, axes = plt.subplots(1, 1, figsize=(8, 4))  # Only one plot needed
    
        # Plot Head vs Capacity
        if has_power and has_efficiency:
            ax1 = axes[0]
        else:
            ax1 = axes

        ax1.scatter(capacities, heads, color="blue", label="Head Data", marker="o")  # Data points only
        ax1.plot(smooth_capacities, head_curve, "b-", label="Head Fit")  # Polynomial curve
        ax1.set_ylabel("Head (m)")
        if chart_title:
            ax1.set_title(f"{chart_title} - {self.fluid.name}")
        else:
            ax1.set_title(f"Pump Performance Curve - {self.fluid.name} Curve")
        ax1.grid()
        ax1.legend()

        if capacity is not None:
            ax1.plot(capacity_value, fitted_head, "bx", markersize=10, label=f"Capacity {capacity_value:.1f} m³/h")
            ax1.axvline(capacity_value, color="gray", linestyle="dashed")  # Vertical dashed line
            ax1.axhline(fitted_head, color="gray", linestyle="dashed")  # Horizontal dashed line

            # Add tick labels
            ax1.text(ax1.get_xlim()[0],
                     fitted_head,
                     f"{fitted_head:.1f}",
                     verticalalignment='center',
                     horizontalalignment='left',
                     fontsize=10, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
            
            if not has_power and not has_efficiency:
                ax1.text(capacity_value,
                         ax1.get_ylim()[0],
                         f"{capacity_value:.1f}",
                         verticalalignment='bottom',
                         horizontalalignment='center',
                         fontsize=10,
                         bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
                

        if has_power and has_efficiency:
            # Power vs Capacity
            powers = np.array([p.breaking_power.to("kW").magnitude for p in self.points])
            power_curve = np.polyval(self.fitter.power_coeffs, smooth_capacities)
            
            ax2 = axes[1]
            ax2.scatter(capacities, powers, color="red", label="Power Data", marker="o")
            ax2.plot(smooth_capacities, power_curve, "r-", label="Power Fit")
            ax2.set_ylabel("Power (kW)")
            ax2.grid()
            ax2.legend()
            ax2.yaxis.set_label_position("right")
            ax2.yaxis.tick_right()

            if capacity is not None:
                ax2.plot(capacity_value, fitted_power, "rx", markersize=10, label=f"Capacity {capacity_value:.1f} m³/h")
                ax2.axvline(capacity_value, color="gray", linestyle="dashed")
                ax2.axhline(fitted_power, color="gray", linestyle="dashed")
                
                # Add tick labels
                ax2.text(ax2.get_xlim()[0],
                         fitted_power,
                         f"{fitted_power:.1f}",
                         verticalalignment='center',
                         horizontalalignment='left',
                         fontsize=10,
                         bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
               
            # Efficiency vs Capacity
            efficiencies = np.array([p.efficiency.to("percent").magnitude for p in self.points])
            efficiency_curve = np.polyval(self.fitter.efficiency_coeffs, smooth_capacities)
            
            ax3 = axes[2]
            ax3.scatter(capacities, efficiencies, color="green", label="Efficiency Data", marker="o")
            ax3.plot(smooth_capacities, efficiency_curve, "g-", label="Efficiency Fit")
            ax3.set_ylabel("Efficiency (%)")
            ax3.set_xlabel("Capacity (m³/h)")
            ax3.grid()
            ax3.legend()

            if capacity is not None:
                ax3.plot(capacity_value, fitted_efficiency, "gx", markersize=10, label=f"Capacity {capacity_value:.1f} m³/h")
                ax3.axvline(capacity_value, color="gray", linestyle="dashed")
                ax3.axhline(fitted_efficiency, color="gray", linestyle="dashed")
    
                # Add tick labels
                ax3.text(capacity_value, ax3.get_ylim()[0], f"{capacity_value:.1f}", verticalalignment='bottom', horizontalalignment='center', fontsize=10, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
                ax3.text(ax3.get_xlim()[0], fitted_efficiency, f"{fitted_efficiency:.1f}", verticalalignment='center', horizontalalignment='left', fontsize=10, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

    
        plt.tight_layout()
        
        if return_io:
            image_stream = io.BytesIO()
            fig.savefig(image_stream, format='png')
            plt.close(fig)
            image_stream.seek(0)
            return image_stream
        plt.show()

        

    def to_speed(self, new_speed: Q_) -> Self:
        """
        Create a new PerformanceCurve with all TestPoints scaled to `new_speed`.

        Uses simplified pump affinity laws:
          - Capacity scales proportionally to speed ratio: Q2 = Q1 * (N2 / N1)
          - Head scales as (N2 / N1)^2 (implicitly handled if you re-compute head).
          - Power scales as (N2 / N1)^3.

        Parameters
        ----------
        new_speed : Q_
            The target speed for all points in this new curve (e.g., Q_(4000, "rpm")).

        Returns
        -------
        PerformanceCurve
            A new PerformanceCurve instance with updated speeds and derived values.

        Raises
        ------
        AttributeError
            If any TestPoint does not have a `speed_of_rotation` attribute.
        """
        new_points = []
        for p in self.points:
            if not hasattr(p, "speed_of_rotation"):
                raise AttributeError(
                    f"TestPoint {p} has no 'speed_of_rotation' attribute."
                )

            # Affinity-laws ratio
            ratio = new_speed / p.speed_of_rotation

            # Scale capacity
            new_capacity = p.capacity * ratio

            # Scales Head
            new_head = p.head * ratio**2

            # Scale power if `breaking_power` exists
            new_breaking_power = (
                p.breaking_power * ratio**3
                if hasattr(p, "breaking_power")
                else None
            )

            # Collect all other attributes except those we want to override
            new_attrs = {}
            new_attrs["_head"] = new_head
            new_attrs["_efficiency"] = p.efficiency
            '''for attr_name, attr_value in p.__dict__.items():
                if attr_name in ["capacity", "speed_of_rotation", "breaking_power"]:
                    continue
                new_attrs[attr_name] = attr_value'''

            # If we had a breaking_power before, include the scaled one
            if new_breaking_power is not None:
                new_attrs["breaking_power"] = new_breaking_power

            # Create a new TestPoint with updated capacity and speed
            new_point = TestPoint(
                fluid=p.fluid,
                capacity=new_capacity,
                speed_of_rotation=new_speed,
                **new_attrs
            )
            new_points.append(new_point)

        return PerformanceCurve(self.fluid, new_points)

    def to_fluid(self, new_fluid: Fluid) -> Self:
        """
        Create a new PerformanceCurve by assigning a new fluid to all TestPoints.

        The capacity, speed, and other dimensional attributes remain the same, but the
        fluid-dependent calculations (head, hydraulic_power, etc.) will reflect the
        properties of the new fluid when accessed.

        Parameters
        ----------
        new_fluid : Fluid
            The new fluid object to assign to all TestPoints.

        Returns
        -------
        PerformanceCurve
            A new PerformanceCurve instance with all TestPoints referencing `new_fluid`.
        """

        new_points = []
        for p in self.points:
            # Collect attributes except the fluid
            new_attrs = {}
            new_attrs["_head"] = p.head
            # If we had a efficiency before, include the scaled one
            if hasattr(p, "_efficiency"):
                new_attrs["_efficiency"] = p.efficiency
                new_attrs["breaking_power"] = quantity_factory(new_fluid.density * p.capacity * p.g * p.head / p.efficiency)
            
            # Create a new TestPoint with the new fluid
            new_point = TestPoint(
                fluid=new_fluid,
                capacity=p.capacity,
                speed_of_rotation=p.speed_of_rotation,
                **new_attrs
            )
            new_points.append(new_point)

        return PerformanceCurve(new_fluid, new_points)
    
    @property
    def test_summary(self) -> str:
        results = []
        for point in self.points:
            results.append([
                f"{point.capacity:0.02f~P}", 
                f"{point.head:0.02f~P}",
                f"{getattr(point, "breaking_power", 0):0.02f~P}" if hasattr(point, "breaking_power") else "N/A",
                f"{getattr(point, "hydraulic_power", 0):0.02f~P}" if hasattr(point, "hydraulic_power") else "N/A",
                f"{getattr(point, "efficiency", 0):0.02f~P}" if hasattr(point, "efficiency") else "N/A",
            ])

        headers = ["Flow", "Head", "Breaking Power", "Hydralic Power", "Efficiency"]
        return tabulate(results, headers=headers, tablefmt="grid")
    
    @property
    def test_data(self) -> Dict:
        data = {
            "Capacity" : [p.capacity for p in self.points],
            "Head" : [p.head for p in self.points],
            "Breaking Power" : [p.breaking_power for p in self.points if hasattr(self.points[0], "breaking_power") ],
            "Hydraulic Power" : [p.hydraulic_power for p in self.points if hasattr(self.points[0], "hydraulic_power") ],
            "Efficiency" : [p.efficiency for p in self.points if hasattr(self.points[0], "efficiency") ],
        }
        return data

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            The number of TestPoints in this collection.
        """
        return len(self.points)

    def __getitem__(self, index: int) -> TestPoint:
        """
        Access an individual TestPoint by index.

        Parameters
        ----------
        index : int
            The index of the desired TestPoint in the `points` list.

        Returns
        -------
        TestPoint
            The corresponding TestPoint.
        """
        return self.points[index]

    def __iter__(self):
        """
        Returns
        -------
        iterator of TestPoint
            An iterator over the TestPoints in this collection.
        """
        return iter(self.points)

    def __repr__(self) -> str:
        """
        Returns
        -------
        str
            A concise string representation of the PerformanceCurve.
        """
        return f"PerformanceCurve(fluid={self.fluid}, points={len(self.points)})"
    

class PerformanceChecker:
    """
    A class to check and validate the performance of a system based on a design point 
    and performance curve.
    """
    def __init__(self, design_point: DesignPoint, performance_curve: PerformanceCurve, acceptable_limits: Optional[Dict[str, float]] = None):
        """
        Initializes the PerformanceChecker with a design point and performance curve.
        
        Parameters
        ----------
        design_point : Any
            The design point containing head, shutoff head, and breaking power values.
        performance_curve : Any
            The performance curve data to be evaluated.
        acceptable_limits : dict, optional
            Predefined acceptable limits for validation (default is None).
        """
        self.design_point = design_point
        self.curve = performance_curve

        self.head_tolerance: float = 0.03  # ±3% of design head
        self.shutoff_tolerance: float = self._get_shutoff_tolerance()
        self.breaking_power_tolerance: float = 0.04

        self._compute_limits()

    def _get_shutoff_tolerance(self) -> float:
        """
        Determines the shutoff head tolerance based on the design point head value.

        Returns
        -------
        float
            The tolerance value for the shutoff head.
        """
        if self.design_point.head.m <= 75:
            return 0.1
        elif self.design_point.head.m <= 300:
            return 0.08
        else:
            return 0.05

    def _compute_limits(self) -> None:
        """
        Computes the acceptable limits for head, shutoff head, and breaking power.
        """
        self.minimum_head = round(self.design_point.head - self.head_tolerance * self.design_point.head, 2)
        self.maximum_head = round(self.design_point.head + self.head_tolerance * self.design_point.head, 2)

        if hasattr(self.design_point, "head_shutoff"):
            self.maximum_head_shutoff = round(self.design_point.head_shutoff + self.shutoff_tolerance * self.design_point.head_shutoff, 2)
            self.minimum_head_shutoff = round(self.design_point.head_shutoff - self.shutoff_tolerance * self.design_point.head_shutoff, 2)

        
        if hasattr(self.design_point, "breaking_power"):
            self.maximum_breaking_power = round(self.design_point.breaking_power + self.breaking_power_tolerance * self.design_point.breaking_power, 2)


    @property
    def acceptable_limits(self) -> Dict[str, float]:
        limits = {
            "Head (min)": self.minimum_head,
            "Head (max)": self.maximum_head,
            "Shutoff Head (min)": self.minimum_head_shutoff,
            "Shutoff Head (max)": self.maximum_head_shutoff,
            "Breaking Power (max)": self.maximum_breaking_power
        }
        return {k: v for k, v in limits.items() if v is not None}

    @property
    def test_summary(self) -> str:
        return self.curve.test_summary
    
    @property
    def check_summary(self) -> str:
        """
        Checks whether the performance curve values fall within the acceptable limits.
        Returns a table summarizing the results.
        """
        results = []
        for point in self.curve:
            head_check = self.minimum_head <= point.head <= self.maximum_head
            shutoff_check = (
                self.minimum_head_shutoff <= point.head <= self.maximum_head_shutoff
                if point.capacity.m < 0.1
                else "N/A"
            )
            power_check = (
                point.breaking_power <= self.maximum_breaking_power
                if hasattr(point, "breaking_power") and hasattr(self, "maximum_breaking_power")
                else "N/A"
            )

            results.append([
                f"{point.capacity:0.02f~P}", 
                f"{point.head:0.02f~P}", 
                head_check,
                f"{point.head:0.02f~P}" if point.capacity.m < 0.1 else "N/A", 
                shutoff_check, 
                f"{getattr(point, "breaking_power", 0):0.02f~P}" if hasattr(point, "breaking_power") else "N/A", 
                power_check
            ])

        headers = ["Flow", "Head", "Head OK", "Shutoff Head", "Shutoff OK", "Breaking Power", "B. Power OK"]
        return tabulate(results, headers=headers, tablefmt="grid")
    
    @property
    def test_summary_with_limits(self) -> str:
        """
        Generates a summary table checking if performance values are within limits.
        
        Returns
        -------
        str
            A formatted table summarizing the check results.
        """
        results = []
        for point in self.curve:                
            results.append([
                f"{point.capacity:0.02f~P}", 
                f"{point.head:0.02f~P}",
                f"{self.minimum_head:0.02f~P}", f"{self.maximum_head:0.02f~P}",
                f"{point.head:0.02f~P}" if point.capacity.m < 0.1 else "N/A",
                f"{self.minimum_head_shutoff:0.02f~P}" if hasattr(self, "minimum_head_shutoff") else "N/A",
                f"{self.maximum_head_shutoff:0.02f~P}" if hasattr(self, "maximum_head_shutoff") else "N/A",
                f"{getattr(point, "breaking_power", 0):0.02f~P}" if hasattr(point, "breaking_power") else "N/A",
                f"{self.maximum_breaking_power:0.02f~P}" if hasattr(self, "maximum_breaking_power") else "N/A"
            ])

        headers = [
            "Flow", "Head", "Head Min", "Head Max", 
            "Shutoff Head", "Shutoff Min", "Shutoff Max",
            "Breaking Power", "Max B. Power"
        ]
        
        
        return tabulate(results, headers=headers, tablefmt="grid")
    

    @property
    def report_summary(self) -> Dict[str, Any]:
        """
        Generates a report summarizing predicted performance values.
        
        Returns
        -------
        dict
            Dictionary containing predicted performance parameters.
        """
        rated_capacity = self.design_point.capacity
        return {
            "Head": [self.curve.predict_head(rated_capacity),
                     self.minimum_head,
                     self.maximum_head],
            "Breaking Power": [self.curve.predict_breaking_power(rated_capacity),
                               "-",
                               self.maximum_breaking_power if hasattr(self, "maximum_breaking_power") else "-"],
            "Efficiency": self.curve.predict_efficiency(rated_capacity),
            "Rated Capacity": self.design_point.capacity
        }