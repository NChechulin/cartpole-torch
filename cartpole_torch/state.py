from dataclasses import dataclass
from typing import Collection

from numpy import pi
from torch import FloatTensor


@dataclass
class State:
    """
    Represents state of the system

    Fields
    ------
    `cart_position` : float
        Position of a cart along X axis.
        Measured in meters.
        When the cart is in the "home state", the position is `0`.

    `pole_angle` : float
        Angle of the pole with respect to inverted Y axis.
        Measured in radians, always belongs to interval `[0, 2*pi)`.
        When the pole is stable (hanging down), the angle is `0`.

    `cart_velocity` : float
        Velocity of the cart along X axis.
        Measured in m/s (meters per second).

    `angular_velocity` : float
        Angular velocity of the pole in counter-clockwise direction.
        Measured in rad/s (radians per second).
    """

    cart_position: float
    pole_angle: float
    cart_velocity: float
    angular_velocity: float

    def __post_init__(self) -> None:
        self.pole_angle %= 2 * pi

    @staticmethod
    def home() -> "State":
        """
        Returns initial state of the system, where all the fields are set to 0.

        Returns
        -------
        State
            initial state of the system
        """
        return State(0, 0, 0, 0)

    @staticmethod
    def target() -> "State":
        """
        Target (ideal) state of the system.

        Returns
        -------
        State
            Ideal state of the system
        """
        return State(
            cart_position=0,
            pole_angle=pi,
            cart_velocity=0,
            angular_velocity=0,
        )

    def as_tensor(self) -> FloatTensor:
        """
        Returns current state as a 1x4 tensor

        Returns
        -------
        FloatTensor
            1x4 Tensor containing `cart_position`,
            `pole_angle`, `cart_velocity` and `angular_velocity`
        """
        return FloatTensor(
            [
                self.cart_position,
                self.pole_angle,
                self.cart_velocity,
                self.angular_velocity,
            ]
        )

    @staticmethod
    def from_collection(arr: Collection[float]) -> "State":
        """
        Creates a State object from an array of length 4.

        Parameters
        ----------
        arr : Collection[float]
            List/Array/Tensor containing `cart_position`, `pole_angle`,
            `cart_velocity` and `angular_velocity`.

        Returns
        -------
        State
            State created from the collection.

        Raises
        ------
        ValueError
            If length of `arr` is not equal to 4.
        """
        if len(arr) != 4:
            raise ValueError("Length of collection should be 4")
        return State(
            cart_position=arr[0],  # type: ignore
            pole_angle=arr[1],  # type: ignore
            cart_velocity=arr[2],  # type: ignore
            angular_velocity=arr[3],  # type: ignore
        )
