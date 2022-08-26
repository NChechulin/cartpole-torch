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
        Measured in m/s (meters per second).

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
        Measured in m/s^2 (meters per second squared).
    """

    pole_length: float = 0.2
    pole_mass: float = 0.118
    gravity: float = 9.807


@dataclass
class SystemConfiguration:
    """
    Contains physical parameters of the system and its' limits.

    Fields
    ------
    `parameters` : SystemParameters
        Physical parameters of the system.

    `limits` : SystemLimits
        Limits of the system, such as maximum position and etc.

    `input_timestep` : float
        Time between adjusting the input of the system.
        We assume the input is the same during the step (i.e. between two
        timesteps).
        Measured in seconds.

    `dynamics_steps_per_input` : int
        Shows how many times we calculate dynamics of the system during
        one timestep.
        We assume the input is the same during the timestep.
        Value of 10 would mean that we update the state of the system
        10 times before adjusting the input.
    """

    parameters: SystemParameters = SystemParameters()
    limits: SystemLimits = SystemLimits()
    input_timestep: float = 1e-2
    dynamics_steps_per_input: int = 10
