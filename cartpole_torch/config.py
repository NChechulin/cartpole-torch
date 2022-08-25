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
