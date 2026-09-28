"""Verificación reproducible de la configuración inicial propuesta para el Caso 1 (Paloma)."""

from math import isclose

from simple_perceptron.dataset import get_training_patterns
from simple_perceptron.model import SimplePerceptron
from simple_perceptron.trainer import train_perceptron


def main() -> None:
    patterns = get_training_patterns()
    model = SimplePerceptron([1.2, 0.8, 2.1], 0.8)
    result = train_perceptron(model, patterns, max_epochs=100)

    expected_targets = tuple(pattern.target for pattern in patterns)

    assert result.converged
    assert result.epochs == 2
    assert all(isclose(a, b, abs_tol=1e-12) for a, b in zip(result.mse_history, [0.5, 0.0]))
    assert all(isclose(a, b, abs_tol=1e-12) for a, b in zip(result.final_weights, [3.2, -1.2, 0.1]))
    assert isclose(result.final_bias, -1.2, abs_tol=1e-12)
    assert result.final_predictions == expected_targets

    print("Verificación correcta para Caso 1 (Aprendizaje de la paloma)")
    print(f"Épocas: {result.epochs}")
    print(f"MSE: {result.mse_history}")
    print(f"Pesos finales: {result.final_weights}")
    print(f"Bias final: {result.final_bias}")
    print(f"Predicciones: {result.final_predictions}")


if __name__ == "__main__":
    main()

