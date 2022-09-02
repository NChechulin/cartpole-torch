from dataclasses import dataclass
from typing import Callable, Optional

import torch
from config import SystemConfiguration
from state import MultiSystemState
from torch import DoubleTensor, IntTensor

CostFunction = Callable[[DoubleTensor], DoubleTensor]


@dataclass
class MultiSystemLearningContext:
    # FIXME: Add docstrings
    states_cost_fn: CostFunction
    inputs_cost_fn: CostFunction
    config: SystemConfiguration
    all_states: DoubleTensor
    batch_state: Optional[MultiSystemState] = None

    def update_batch(self, batch_size: int) -> None:
        """
        Generates a new batch and updates the batch multistate.

        Parameters
        ----------
        batch_size : int
            Size of the batch.

        Raises
        ------
        ValueError
            If `batch_size` was less than 1 or greater than
            the total number of states.
        """
        total_states = self.all_states.shape[1]

        if not (0 < batch_size <= total_states):
            raise ValueError("Invalid batch size")

        # Generate 1xK tensor with values from [0, 1)
        rand = torch.rand(size=batch_size)  # type: ignore

        # Multiply the numbers so they are integers from [0, total_states)
        batch: IntTensor = (rand * total_states).int()  # type: ignore

        self.batch_state = MultiSystemState.from_batch(self.all_states, batch)
