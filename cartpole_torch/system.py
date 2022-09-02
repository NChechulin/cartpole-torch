from dataclasses import dataclass

from config import SystemConfiguration
from state import State
from torch import DoubleTensor, cos, sin

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
        cur_st: DoubleTensor = self.current_state.as_tensor()
        steps: int = self.config.dynamics_steps_per_input
        # Delta time
        dt: float = self.config.input_timestep / steps

        grav: float = self.config.parameters.gravity  # Gravitational constant
        pole_len: float = self.config.parameters.pole_length

        def __compute_derivative(state: DoubleTensor):
            # FIXME: Use 2 pre-allocated arrays instead of creating new ones
            delta_state = DoubleTensor([0, 0, 0, 0])
            ang = state[1]  # 1x1 Tensor with angle
            u = best_input
            delta_state[0] = state[2]
            delta_state[1] = state[3]
            delta_state[2] = u
            delta_state[3] = -1.5 / pole_len * (u * cos(ang) + grav * sin(ang))
            return delta_state

        for _ in range(steps):
            # Evaluate derivatives
            t1 = __compute_derivative(cur_st)
            t2 = __compute_derivative(cur_st + t1 * dt)  # type: ignore
            cur_st += (t1 + t2) / 2 * dt  # type: ignore

        self.history.add_entry(
            timestamp=self.simulation_time,
            current_input=best_input,
            state=self.current_state,
        )
        self.current_state = State.from_collection(cur_st)  # type: ignore
        self.simulation_time += self.config.input_timestep
