from dataclasses import dataclass

from config import SystemConfiguration
from state import State
from torch import FloatTensor, cos, sin


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

    def advance_to(self, target_time: float) -> None:
        """
        Advances the system to a given moment in time.

        Parameters
        ----------
        target_time : float
            New time of the system.
            Should be greater than `simulation_time`.

        Raises
        ------
        ValueError
            If target time is smaller than `simulation_time`.
        """
        if target_time < self.simulation_time:
            raise ValueError("Target time should be greater than current time")

        time_delta: float = target_time - self.simulation_time
        steps: int = round(time_delta / self.config.input_timestep)

        for _ in range(steps):
            self.advance_one_step()

    def advance_one_step(self) -> None:
        """
        Advances the system one step further.
        One step equals `config.input_timestep` seconds.
        """

        # Current state
        cur_st: FloatTensor = self.current_state.as_tensor()
        steps: int = self.config.dynamics_steps_per_input
        # Delta time
        dt: float = self.config.input_timestep / steps

        # FIXME: add real value
        inp: float = 0  # Optimal input to the system
        grav: float = self.config.parameters.gravity  # Gravitational constant
        pole_len: float = self.config.parameters.pole_length

        for _ in range(steps):
            # Evaluate derivatives
            delta_state: FloatTensor = FloatTensor(
                [
                    cur_st[2] * dt,
                    cur_st[3] * dt,
                    inp * dt,
                    -(inp * cos(cur_st[1]) + grav * sin(cur_st[1])) / pole_len * dt,
                ]
            )
            cur_st += delta_state  # type: ignore
        self.current_state = State.from_collection(cur_st)  # type: ignore
