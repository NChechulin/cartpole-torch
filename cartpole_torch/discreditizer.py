from dataclasses import dataclass

import torch
from config import SystemConfiguration
from torch import DoubleTensor


@dataclass
class Discreditizer:
    config: SystemConfiguration

    cart_accelerations: DoubleTensor = DoubleTensor()
    __all_states: DoubleTensor = DoubleTensor()

    def __post_init__(self) -> None:
        """
        Generates all states from configuration
        """
        xs = torch.linspace(  # type: ignore
            start=-self.config.limits.max_abs_position,
            end=self.config.limits.max_abs_position,
            steps=self.config.discretization.cart_position,
            dtype=torch.float64,
        )
        thetas = torch.linspace(  # type: ignore
            start=0.0,
            end=2 * torch.pi,
            steps=self.config.discretization.pole_angle,
            dtype=torch.float64,
        )
        xdots = torch.linspace(  # type: ignore
            start=-self.config.limits.max_abs_velocity,
            end=self.config.limits.max_abs_velocity,
            steps=self.config.discretization.cart_velocity,
            dtype=torch.float64,
        )
        thetadots = torch.linspace(  # type: ignore
            start=-self.config.limits.max_abs_angular_velocity,
            end=self.config.limits.max_abs_angular_velocity,
            steps=self.config.discretization.pole_angular_velocity,
            dtype=torch.float64,
        )
        self.cart_accelerations = torch.linspace(  # type: ignore
            start=-self.config.limits.max_abs_acceleration,
            end=self.config.limits.max_abs_acceleration,
            steps=self.config.discretization.cart_acceleration,
            dtype=torch.float64,
        )

        # Generate all possible states tensor
        xs, thetas, xdots, thetadots = torch.meshgrid(
            [
                xs,
                thetas,
                xdots,
                thetadots,
            ],
            indexing="ij",
        )

        self.__all_states = torch.vstack(  # type: ignore
            [
                xs.flatten(),
                thetas.flatten(),
                xdots.flatten(),
                thetadots.flatten(),
            ]
        )

    @property
    def space_size(self) -> int:
        """
        Returns the total number of states

        Returns
        -------
        int
        """
        return self.__all_states.shape[1]

    @property
    def cart_positions(self) -> DoubleTensor:
        return self.__all_states[0]  # type: ignore

    @property
    def pole_angles(self) -> DoubleTensor:
        return self.__all_states[1]  # type: ignore

    @property
    def cart_velocities(self) -> DoubleTensor:
        return self.__all_states[2]  # type: ignore

    @property
    def pole_angular_velocities(self) -> DoubleTensor:
        return self.__all_states[3]  # type: ignore

    @property
    def all_states(self) -> DoubleTensor:
        return self.__all_states
