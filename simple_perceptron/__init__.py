"""Solución modular del Caso de Estudio 1 - Perceptrón Simple (Aprendizaje Paloma)."""

from .dataset import Pattern, get_training_patterns
from .model import SimplePerceptron
from .trainer import TrainingResult, train_perceptron

__all__ = [
    "Pattern",
    "get_training_patterns",
    "SimplePerceptron",
    "TrainingResult",
    "train_perceptron",
]
