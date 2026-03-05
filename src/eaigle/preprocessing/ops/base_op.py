
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

class BaseOp(ABC):
    

    @abstractmethod
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
