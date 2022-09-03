"""
This module contains 3 classes related to keeping record of systems states:
- `HistoryTensorFields` is a simple enum which assigns
    human-readable names to tensor rows.
- `HistoryEntry` which represents state of the system at a certain
    point in time before applying a specified input.
- `SystemHistory` which stores `HistoryEntries` and provides convenient
    methods to explore data.
"""


from dataclasses import dataclass, field
from enum import IntEnum, auto

import torch
from config import SystemParameters
from state import State
from torch import DoubleTensor, Tensor, cos


class HistoryTensorFields(IntEnum):
    """
    Maps Tensor rows into human-readable constants and vice-versa.
    """

    TIMESTAMP = 0
    INPUT = auto()
    CART_POSITION = auto()
    POLE_ANGLE = auto()
    CART_VELOCITY = auto()
    POLE_ANGULAR_VELOCITY = auto()


@dataclass
class HistoryEntry:
    """
    Represents state of the system at a certain point in time before
    applying a specified input.

    Fields
    ------
    `timestamp` : float
        Time (in seconds) since the start of the simulation.

    `current_input` : float
        Input to the system during current step.
        Measured in m/s^2 (meters per second squared).

    `state` : State
        State of the system at the end of the current step.
    """

    timestamp: float
    current_input: float
    state: State

    def as_tensor(self) -> DoubleTensor:
        """
        Represent history entry as a 1x6 tensor.

        Returns
        -------
        DoubleTensor
            Tensor of length 6 containing the following columns:
            - `timestamp` - time since start of the simulation (in seconds)
            - `input` - input to the system (float, m/s^2)
            - `position` - position of the cart (float, m)
            - `angle` - angle of the pole (float, rad)
            - `velocity` - velocity of the cart (float, m/s)
            - `angular_velocity` - angular velocity of the pole (float, rad/s)
        """

        pos, angle, velocity, ang_velocity = self.state.as_tensor()
        return DoubleTensor(
            [
                self.timestamp,
                self.current_input,
                pos,
                angle,
                velocity,
                ang_velocity,
            ]
        )


@dataclass
class SystemHistory:
    """
    A smart container for history entries which provides convenient
    methods to explore data.
    """

    entries: list[HistoryEntry] = field(default_factory=list)

    def add_entry(
        self,
        timestamp: float,
        current_input: float,
        state: State,
    ) -> None:
        """
        Adds an entry to history

        Parameters
        ----------
        `timestamp` : float
            Time (in seconds) since the start of the simulation.

        `current_input` : float
            Input to the system during current step.
            Measured in m/s^2 (meters per second squared).

        `state` : State
            Current state of the system (after the step).
        """
        # FIXME: Rewrite using concat (should be faster)
        self.entries.append(HistoryEntry(timestamp, current_input, state))

    def as_tensor(self) -> DoubleTensor:
        """
        Converts history to a Nx6 tensor

        Returns
        -------
        Tensor
            Nx6 tensor containing the following columns:
            - `timestamp` - time since start of the simulation (in seconds)
            - `input` - input to the system (float, m/s^2)
            - `cart_position` - position of the cart (float, m)
            - `pole_angle` - angle of the pole (float, rad)
            - `cart_velocity` - velocity of the cart (float, m/s)
            - `pole_angular_velocity` - angular velocity of the
            pole (float, rad/s)
        """
        # FIXME: use concat with axis
        res = [entry.as_tensor() for entry in self.entries]  # type: ignore
        res: DoubleTensor = torch.cat(res)  # type: ignore

        return res.reshape([-1, 6])  # type: ignore

    def timestamps(self) -> Tensor:
        """
        Returns
        -------
        Tensor
            1xN Tensor with timestamps.
        """
        # FIXME: ineffective implementation (.as_tensor() is costy)
        return self.as_tensor()[:, HistoryTensorFields.TIMESTAMP]

    def inputs(self) -> Tensor:
        """
        Returns
        -------
        Tensor
            1xN Tensor with inputs to the system.
        """
        # FIXME: ineffective implementation (.as_tensor() is costy)
        return self.as_tensor()[:, HistoryTensorFields.INPUT]

    def cart_positions(self) -> Tensor:
        """
        Returns
        -------
        Tensor
            1xN Tensor with positions of the cart.
        """
        # FIXME: ineffective implementation (.as_tensor() is costy)
        return self.as_tensor()[:, HistoryTensorFields.CART_POSITION]

    def pole_angles(self) -> Tensor:
        """
        Returns
        -------
        Tensor
            1xN Tensor with angles of the pole.
        """
        # FIXME: ineffective implementation (.as_tensor() is costy)
        return self.as_tensor()[:, HistoryTensorFields.POLE_ANGLE]

    def cart_velocities(self) -> Tensor:
        """
        Returns
        -------
        Tensor
            1xN Tensor with velocities of the cart.
        """
        # FIXME: ineffective implementation (.as_tensor() is costy)
        return self.as_tensor()[:, HistoryTensorFields.CART_VELOCITY]

    def pole_angular_velocities(self) -> Tensor:
        """
        Returns
        -------
        Tensor
            1xN Tensor with angles of the pole.
        """
        # FIXME: ineffective implementation (.as_tensor() is costy)
        return self.as_tensor()[:, HistoryTensorFields.POLE_ANGULAR_VELOCITY]

    def total_energies(self, config: SystemParameters) -> Tensor:
        """
        Returns
        -------
        Tensor
            1xN Tensor with total energies at each step.
        """
        # FIXME: ineffective implementation (.as_tensor() is costy)

        kin_cart = torch.zeros(
            len(self.entries)
        )  # FIXME: add actual cart mass to config
        pot_cart = torch.zeros(len(self.entries))

        pot_pole = (
            config.pole_mass
            * config.gravity
            * config.pole_length
            / 2
            * (1 - cos(self.pole_angles()))
        )

        velocities = self.cart_velocities()
        angular_velocities = self.pole_angular_velocities()
        kin_pole = (config.pole_mass / 2) * (
            velocities**2
            + ((config.pole_length**2) * (angular_velocities**2)) / 3
            + (
                config.pole_length
                * velocities
                * angular_velocities
                * cos(self.pole_angles())
            )
        )
        return kin_cart + pot_cart + kin_pole + pot_pole

    # TODO: add actual smart methods
    # like plotting / generating animations / smth else
