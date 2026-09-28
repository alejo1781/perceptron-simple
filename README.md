# Caso de Estudio 1 - Aprendizaje de una Paloma (Perceptrón Simple)

Solución modular en Python con interfaz gráfica Tkinter para simular el aprendizaje de una paloma en la selección del pulsador adecuado para conseguir comida, a partir del estado de dos pulsadores y de la elección realizada por la paloma.

## Regla del problema

- **Entradas bipolares {-1, +1}**:
  - `x1`: Estado del pulsador izquierdo (-1 = apagado, +1 = encendido)
  - `x2`: Estado del pulsador derecho (-1 = apagado, +1 = encendido)
  - `x3`: Elección de la paloma (-1 = pulsador izquierdo, +1 = pulsador derecho)
- **Salida deseada T**:
  - `+1`: Éxito (la paloma consigue comida)
  - `-1`: Fracaso (la paloma no consigue comida)

## Algoritmo de entrenamiento

Para cada patrón en cada época:

```text
a = W · X + b
Y = +1 si a >= 0; en otro caso Y = -1  (hardlims)
e = T - Y
W <- W + e·X
b <- b + e
```

Al terminar la época se calcula el error global (MSE) con los errores de esa misma pasada:

```text
MSE = (1/N) * sum(e(k)^2)
```

El entrenamiento finaliza cuando `MSE == 0`.

## Estructura del proyecto

```text
app.py
verify_training.py
requirements.txt
README.md
simple_perceptron/
    __init__.py
    dataset.py
    model.py
    trainer.py
    plotting.py
    gui.py
```

## Instalación

```bash
python -m pip install -r requirements.txt
```

## Ejecución

```bash
python app.py
```

## Verificación automática

```bash
python verify_training.py
```

Con los valores iniciales propuestos del Caso 1:

```text
W0 = (1.2, 0.8, 2.1)
b0 = 0.8
```

Se obtiene:

```text
MSE por época = [0.5, 0.0]
Convergencia = Época 2
Wfinal = (3.2, -1.2, 0.1)
bfinal = -1.2
```

La interfaz gráfica permite cambiar los pesos iniciales, el bias y el número máximo de épocas antes de ejecutar la simulación.
