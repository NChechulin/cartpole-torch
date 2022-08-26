"""
Defines configuration of the system - its' limits and physical parameters
"""
from dataclasses import dataclass


@dataclass
class SystemLimits:
    """
    Represents system limits

    Fields
    ------
    `max_abs_position` : float
        Maximum absolute position of the cart, so the real
        position belongs to (-max_abs_position, max_abs_position).
        Measured in meters.

    `max_abs_velocity` : float
        Maximum absolute velocity of the cart.
        Measured in m/s (meters per second)

    `max_abs_acceleration` : float
        Maximum absolute acceleration of the cart.
        Measured in m/s^2 (meters per second squared).
    """

    max_abs_position: float = 0.25
    max_abs_velocity: float = 25.0
    max_abs_acceleration: float = 7


@dataclass
class SystemParameters:
    """
    Represents physical parameters of the system

    Fields
    ------
    `pole_length` : float
        Length of the pole in meters.

    `pole_mass` : float
        Mass of the pole in kg.

    `gravity` : float
        Gravitational constant, shows the gravitational pull.
        Measured in m/s^2 (meters per second squared)
    """

    pole_length: float = 0.2
    pole_mass: float = 0.118
    gravity: float = 9.807


@dataclass
class SystemConfiguration:
    """
    Contains physical parameters of the system and its' limits
    """

    parameters: SystemParameters = SystemParameters()
    limits: SystemLimits = SystemLimits()
