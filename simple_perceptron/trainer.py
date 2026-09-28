"""Algoritmo de entrenamiento supervisado del Perceptrón Simple."""

from dataclasses import dataclass
from typing import List, Sequence

from .dataset import Pattern
from .model import SimplePerceptron


@dataclass(frozen=True)
class PatternStep:
    epoch: int
    pattern_name: str
    inputs: tuple[int, int, int]
    target: int
    weights_before: tuple[float, float, float]
    bias_before: float
    net: float
    output: int
    error: int
    squared_error: int
    weights_after: tuple[float, float, float]
    bias_after: float

    @property
    def updated(self) -> bool:
        return self.error != 0


@dataclass(frozen=True)
class EpochMetric:
    epoch: int
    mse: float
    sse: float
    misclassified_during_pass: int
    updates: int
    final_weights: tuple[float, float, float]
    final_bias: float


@dataclass(frozen=True)
class TrainingResult:
    converged: bool
    epochs: int
    steps: tuple[PatternStep, ...]
    metrics: tuple[EpochMetric, ...]
    final_weights: tuple[float, float, float]
    final_bias: float
    final_predictions: tuple[int, ...]

    @property
    def mse_history(self) -> List[float]:
        return [metric.mse for metric in self.metrics]


def train_perceptron(
    model: SimplePerceptron,
    patterns: Sequence[Pattern],
    max_epochs: int = 100,
) -> TrainingResult:
    """Entrena el perceptrón siguiendo el criterio global MSE por época.

    Importante: el MSE se calcula con los errores e(k)=T(k)-Y(k) obtenidos
    durante la misma pasada secuencial de entrenamiento. No se realiza una
    segunda evaluación para decidir la parada.

    MSE_epoca = (1/N) * sum(e(k)^2)

    El entrenamiento se detiene cuando MSE_epoca == 0, lo que implica que
    todos los patrones fueron clasificados correctamente durante una época
    completa y, por tanto, no hubo actualizaciones efectivas.
    """
    if not patterns:
        raise ValueError("El conjunto de patrones no puede estar vacío.")
    if max_epochs <= 0:
        raise ValueError("max_epochs debe ser mayor que cero.")

    all_steps: List[PatternStep] = []
    metrics: List[EpochMetric] = []
    converged = False

    for epoch in range(1, max_epochs + 1):
        squared_error_sum = 0.0
        misclassified = 0
        updates = 0

        for pattern in patterns:
            weights_before = tuple(model.copy_weights())
            bias_before = model.bias

            net = model.net_input(pattern.inputs)
            output = model.hardlims(net)
            error = pattern.target - output
            squared_error = error ** 2

            squared_error_sum += squared_error
            if error != 0:
                misclassified += 1
                updates += 1

            # Se aplica la regla completa; con e=0 no produce cambio.
            model.update(pattern.inputs, error)

            all_steps.append(
                PatternStep(
                    epoch=epoch,
                    pattern_name=pattern.name,
                    inputs=pattern.inputs,
                    target=pattern.target,
                    weights_before=weights_before,
                    bias_before=bias_before,
                    net=net,
                    output=output,
                    error=error,
                    squared_error=squared_error,
                    weights_after=tuple(model.copy_weights()),
                    bias_after=model.bias,
                )
            )

        mse = squared_error_sum / len(patterns)
        metrics.append(
            EpochMetric(
                epoch=epoch,
                mse=mse,
                sse=squared_error_sum,
                misclassified_during_pass=misclassified,
                updates=updates,
                final_weights=tuple(model.copy_weights()),
                final_bias=model.bias,
            )
        )

        if mse == 0.0:
            converged = True
            break

    final_predictions = tuple(model.predict(pattern.inputs) for pattern in patterns)

    return TrainingResult(
        converged=converged,
        epochs=len(metrics),
        steps=tuple(all_steps),
        metrics=tuple(metrics),
        final_weights=tuple(model.copy_weights()),
        final_bias=model.bias,
        final_predictions=final_predictions,
    )
