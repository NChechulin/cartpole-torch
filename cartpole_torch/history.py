from dataclasses import dataclass, field

import torch
from state import State
from torch import FloatTensor


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

    def as_tensor(self) -> FloatTensor:
        """
        Represent history entry as a 1x6 tensor.

        Returns
        -------
        FloatTensor
            Tensor of length 6 containing the following columns:
            - `timestamp` - time since start of the simulation (in seconds)
            - `input` - input to the system (float, m/s^2)
            - `position` - position of the cart (float, m)
            - `angle` - angle of the pole (float, rad)
            - `velocity` - velocity of the cart (float, m/s)
            - `angular_velocity` - angular velocity of the pole (float, rad/s)
        """

        pos, angle, velocity, ang_speed = self.state.as_tensor()
        return FloatTensor(
            [
                self.timestamp,
                self.current_input,
                pos,
                angle,
                velocity,
                ang_speed,
            ]
        )


@dataclass
class SystemHistory:
    """
    A smart container for history entries
    """

    entries: list[HistoryEntry] = field(default_factory=list)

    def as_tensor(self) -> FloatTensor:
        """
        Converts history to an Nx6 tensor

        Returns
        -------
        Tensor
            Nx6 tensor containing the following columns:
            - `timestamp` - time since start of the simulation (in seconds)
            - `input` - input to the system (float, m/s^2)
            - `position` - position of the cart (float, m)
            - `angle` - angle of the pole (float, rad)
            - `velocity` - velocity of the cart (float, m/s)
            - `angular_velocity` - angular velocity of the pole (float, rad/s)
        """
        res = [entry.as_tensor() for entry in self.entries]  # type: ignore
        res: FloatTensor = torch.cat(res)  # type: ignore

        return res.reshape([-1, 6])  # type: ignore

    # TODO: add actual smart methods
