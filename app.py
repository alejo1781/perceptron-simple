"""Punto de entrada único de la aplicación."""

from simple_perceptron.gui import PerceptronApp


def main() -> None:
    app = PerceptronApp()
    app.mainloop()


if __name__ == "__main__":
    main()
