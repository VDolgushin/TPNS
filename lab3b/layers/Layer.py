from abc import abstractmethod, ABC
import numpy as np


class Layer(ABC):
    @abstractmethod
    def calculate_output(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def back_propagation(self, do: np.ndarray, learning_rate) -> np.ndarray:
        pass
