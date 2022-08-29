from dataclasses import dataclass

from config import SystemConfiguration
from state import State
from torch import FloatTensor, cos, sin

from cartpole_torch.history import SystemHistory


@dataclass
class CartPoleSystem:
    """
    Defines the system as a whole

    Fields
    ------
    `config` : SystemConfiguration
        Physical configuration of the system.

    `current_state` : State
        Current state of the system.
        Changes over time.

    `target_state` : State
        State we want the system to be in.
        Should remain constant.

    `simulation_time` : float
        Time (in seconds) since the start of the simulation.
    """

    config: SystemConfiguration = SystemConfiguration()
    current_state: State = State.home()
    # TODO: remove field and use State.target() to ensure immutability
    # TODO: think of cases when the field should be mutable
    target_state: State = State.target()
    simulation_time: float = 0.0
    history: SystemHistory = SystemHistory()

    def advance_to(self, target_time: float, best_input: float) -> None:
        """
        Advances the system to a given moment in time.

        Parameters
        ----------
        `target_time` : float
            New time of the system.
            Should be greater than `simulation_time`.
        `best_input` : float
            Input to the system (in m/s^2).
            Will be removed.

        Raises
        ------
        ValueError
            If target time is smaller than `simulation_time`.
        """
        if target_time < self.simulation_time:
            raise ValueError("Target time should be greater than current time")
        # FIXME: Remove best input from args and calculate it each time
        while self.simulation_time < target_time:
            self.advance_one_step(best_input)

    def advance_one_step(self, best_input: float) -> None:
        """
        Advances the system one step further.
        One step equals `config.input_timestep` seconds.
        """
        # Current state
        cur_st: FloatTensor = self.current_state.as_tensor()
        steps: int = self.config.dynamics_steps_per_input
        # Delta time
        dt: float = self.config.input_timestep / steps

        grav: float = self.config.parameters.gravity  # Gravitational constant
        pole_len: float = self.config.parameters.pole_length

        for _ in range(steps):
            # Evaluate derivatives
            ang = cur_st[1]  # 1x1 Tensor with angle
            delta_state: FloatTensor = FloatTensor(
                [
                    cur_st[2] * dt,
                    cur_st[3] * dt,
                    best_input * dt,
                    -(best_input * cos(ang) + grav * sin(ang)) / pole_len * dt,
                ]
            )
            cur_st += delta_state  # type: ignore

        self.current_state = State.from_collection(cur_st)  # type: ignore
        self.history.add_entry(
            timestamp=self.simulation_time,
            current_input=best_input,
            state=self.current_state,
        )
        self.simulation_time += self.config.input_timestep
