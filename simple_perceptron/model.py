"""Modelo matemático de una neurona Perceptrón Simple bipolar."""

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass
class SimplePerceptron:
    """Perceptrón simple con tres entradas y función hardlims.

    La regla de aprendizaje utilizada es:
        e = T - Y
        W(nuevo) = W(actual) + e * X
        b(nuevo) = b(actual) + e

    La tasa de aprendizaje efectiva es 1, igual que en el algoritmo de referencia.
    """

    weights: List[float]
    bias: float

    def __post_init__(self) -> None:
        if len(self.weights) != 3:
            raise ValueError("El perceptrón debe tener exactamente tres pesos.")
        self.weights = [float(value) for value in self.weights]
        self.bias = float(self.bias)

    def net_input(self, inputs: Sequence[int]) -> float:
        if len(inputs) != 3:
            raise ValueError("Se requieren exactamente tres entradas.")
        return sum(w * x for w, x in zip(self.weights, inputs)) + self.bias

    @staticmethod
    def hardlims(net: float) -> int:
        """Función de activación bipolar: +1 si net >= 0; -1 en otro caso."""
        return 1 if net >= 0 else -1

    def predict(self, inputs: Sequence[int]) -> int:
        return self.hardlims(self.net_input(inputs))

    def update(self, inputs: Sequence[int], error: int) -> None:
        """Aplica la regla de aprendizaje del perceptrón."""
        self.weights = [w + error * x for w, x in zip(self.weights, inputs)]
        self.bias += error

    def copy_weights(self) -> List[float]:
        return list(self.weights)
