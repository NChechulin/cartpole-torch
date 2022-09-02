from dataclasses import dataclass, field
from enum import IntEnum, auto

import torch
from state import State
from torch import DoubleTensor, Tensor, cos

from cartpole_torch.config import SystemParameters


class HistoryTensorFields(IntEnum):
    TIMESTAMP = 0
    INPUT = auto()
    CART_POSITION = auto()
    POLE_ANGLE = auto()
    CART_VELOCITY = auto()
    POLE_ANGULAR_VELOCITY = auto()


@dataclass
class HistoryEntry:
    """
    _summary_

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
    A smart container for history entries
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

        m_p = config.pole_mass
        l_p = config.pole_length
        g = config.gravity

        pot_pole = m_p * g * l_p / 2 * (1 - cos(self.pole_angles()))

        vs = self.cart_velocities()
        ws = self.pole_angular_velocities()
        kin_pole = (m_p / 2) * (
            vs**2
            + ((l_p**2) * (ws**2)) / 3
            + l_p * vs * ws * cos(self.pole_angles())
        )
        return kin_cart + pot_cart + kin_pole + pot_pole

    # TODO: add actual smart methods
