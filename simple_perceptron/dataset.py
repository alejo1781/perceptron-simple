"""Definición del conjunto de patrones bipolares del Caso 1: Aprendizaje de la paloma."""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Pattern:
    """Un patrón supervisado de entrada/salida para el aprendizaje de la paloma."""

    name: str
    inputs: Tuple[int, int, int]
    target: int

    @property
    def result_text(self) -> str:
        """Resultado del intento de la paloma: Éxito (+1) o Fracaso (-1)."""
        return "Éxito" if self.target == 1 else "Fracaso"

    @property
    def diagnosis(self) -> str:
        """Compatibilidad con la propiedad previa."""
        return self.result_text


def get_training_patterns() -> List[Pattern]:
    """Genera los 8 patrones bipolares del Caso 1:

    Patrón | x1 (Pulsador izq) | x2 (Pulsador der) | x3 (Elección) | T
    P1     |         1         |         1         |       1       |  1
    P2     |         1         |         1         |      -1       |  1
    P3     |         1         |        -1         |       1       |  1
    P4     |        -1         |         1         |       1       | -1
    P5     |         1         |        -1         |      -1       |  1
    P6     |        -1         |         1         |      -1       | -1
    P7     |        -1         |        -1         |      -1       | -1
    P8     |        -1         |        -1         |       1       | -1
    """
    raw_patterns = [
        ("P1", (1, 1, 1), 1),
        ("P2", (1, 1, -1), 1),
        ("P3", (1, -1, 1), 1),
        ("P4", (-1, 1, 1), -1),
        ("P5", (1, -1, -1), 1),
        ("P6", (-1, 1, -1), -1),
        ("P7", (-1, -1, -1), -1),
        ("P8", (-1, -1, 1), -1),
    ]
    return [Pattern(name, inputs, target) for name, inputs, target in raw_patterns]

