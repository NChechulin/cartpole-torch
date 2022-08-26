from dataclasses import dataclass

from config import SystemConfiguration
from state import State


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

        raise NotImplementedError

    def advance_one_step(self) -> None:
        """
        Advances the system one step further.
        One step equals `config.input_timestep` seconds.
        """

        raise NotImplementedError
