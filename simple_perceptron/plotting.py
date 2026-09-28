"""Funciones de visualización de las métricas de entrenamiento."""

from matplotlib.axes import Axes

from .trainer import TrainingResult


def plot_mse(ax: Axes, result: TrainingResult) -> None:
    """Dibuja la evolución del MSE global por época en un Axes existente."""
    ax.clear()
    epochs = [metric.epoch for metric in result.metrics]
    mse_values = [metric.mse for metric in result.metrics]

    ax.plot(epochs, mse_values, marker="o", linewidth=2)
    ax.set_title("Evolución del MSE global por época")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE global")
    ax.set_xticks(epochs)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    for epoch, mse in zip(epochs, mse_values):
        ax.annotate(
            f"{mse:.3g}",
            (epoch, mse),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )
