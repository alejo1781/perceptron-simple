"""Interfaz gráfica Tkinter de la simulación del Perceptrón Simple - Caso 1: Aprendizaje de la Paloma."""

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .dataset import get_training_patterns
from .model import SimplePerceptron
from .plotting import plot_mse
from .trainer import TrainingResult, train_perceptron


DEFAULT_WEIGHTS = (1.2, 0.8, 2.1)
DEFAULT_BIAS = 0.8
DEFAULT_MAX_EPOCHS = 100


class PerceptronApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Caso de Estudio 1 - Perceptrón Simple (Aprendizaje Paloma)")
        self.geometry("1180x760")
        self.minsize(980, 680)

        self.patterns = get_training_patterns()
        self.result: Optional[TrainingResult] = None
        self.model: Optional[SimplePerceptron] = None

        self._configure_style()
        self._build_layout()
        self._load_pattern_table()
        self._show_algorithm_text()
        self._show_initial_graph()

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Sub.TLabel", font=("Segoe UI", 10))
        style.configure("Result.TLabel", font=("Segoe UI", 10, "bold"))

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        ttk.Label(
            root,
            text="Caso 1: Aprendizaje de una paloma para la manera de conseguir comida",
            style="Title.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            root,
            text=(
                "Entradas x1 (Pulsador izq), x2 (Pulsador der), x3 (Elección paloma) en representación bipolar {-1, +1}.\n"
                "Salida deseada T: +1 (Éxito / Consigue comida), -1 (Fracaso / No consigue comida)."
            ),
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(2, 10))

        params = ttk.LabelFrame(root, text="Parámetros iniciales (Caso 1)", padding=10)
        params.pack(fill="x", pady=(0, 10))

        self.w1_var = tk.StringVar(value=str(DEFAULT_WEIGHTS[0]))
        self.w2_var = tk.StringVar(value=str(DEFAULT_WEIGHTS[1]))
        self.w3_var = tk.StringVar(value=str(DEFAULT_WEIGHTS[2]))
        self.bias_var = tk.StringVar(value=str(DEFAULT_BIAS))
        self.max_epochs_var = tk.StringVar(value=str(DEFAULT_MAX_EPOCHS))

        fields = [
            ("w1", self.w1_var),
            ("w2", self.w2_var),
            ("w3", self.w3_var),
            ("Bias b", self.bias_var),
            ("Máx. épocas", self.max_epochs_var),
        ]
        for column, (label, variable) in enumerate(fields):
            ttk.Label(params, text=label).grid(row=0, column=2 * column, padx=(0, 4))
            ttk.Entry(params, width=10, textvariable=variable).grid(
                row=0, column=2 * column + 1, padx=(0, 12)
            )

        ttk.Button(params, text="Entrenar", command=self._train).grid(row=0, column=10, padx=4)
        ttk.Button(params, text="Restablecer", command=self._reset).grid(row=0, column=11, padx=4)

        self.status_var = tk.StringVar(value="Listo para entrenar.")
        ttk.Label(params, textvariable=self.status_var, style="Result.TLabel").grid(
            row=1, column=0, columnspan=12, sticky="w", pady=(8, 0)
        )

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        self.tab_patterns = ttk.Frame(self.notebook, padding=10)
        self.tab_training = ttk.Frame(self.notebook, padding=10)
        self.tab_graph = ttk.Frame(self.notebook, padding=10)
        self.tab_algorithm = ttk.Frame(self.notebook, padding=10)
        self.tab_patient = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.tab_patterns, text="Patrones")
        self.notebook.add(self.tab_training, text="Entrenamiento")
        self.notebook.add(self.tab_graph, text="Error por época")
        self.notebook.add(self.tab_algorithm, text="Algoritmo")
        self.notebook.add(self.tab_patient, text="Probar paloma")

        self._build_patterns_tab()
        self._build_training_tab()
        self._build_graph_tab()
        self._build_algorithm_tab()
        self._build_patient_tab()

    def _build_patterns_tab(self) -> None:
        columns = ("patron", "x1", "x2", "x3", "target", "pred")
        self.pattern_tree = ttk.Treeview(self.tab_patterns, columns=columns, show="headings", height=12)
        headings = {
            "patron": "Patrón",
            "x1": "Pulsador izq x1",
            "x2": "Pulsador der x2",
            "x3": "Elección paloma x3",
            "target": "Deseada T",
            "pred": "Salida final Y",
        }
        widths = {"patron": 80, "x1": 130, "x2": 130, "x3": 150, "target": 100, "pred": 120}
        for column in columns:
            self.pattern_tree.heading(column, text=headings[column])
            self.pattern_tree.column(column, width=widths[column], anchor="center")
        self.pattern_tree.pack(fill="both", expand=True)

        ttk.Label(
            self.tab_patterns,
            text="Codificación bipolar: -1 = Apagado / Pulsador izq; +1 = Encendido / Pulsador der. T=+1 Éxito; T=-1 Fracaso.",
        ).pack(anchor="w", pady=(8, 0))

    def _build_training_tab(self) -> None:
        frame = ttk.Frame(self.tab_training)
        frame.pack(fill="both", expand=True)
        self.training_text = tk.Text(frame, wrap="none", font=("Consolas", 10))
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=self.training_text.yview)
        xscroll = ttk.Scrollbar(frame, orient="horizontal", command=self.training_text.xview)
        self.training_text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.training_text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

    def _build_graph_tab(self) -> None:
        self.figure = Figure(figsize=(8.5, 5.2), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.tab_graph)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_algorithm_tab(self) -> None:
        self.algorithm_text = tk.Text(self.tab_algorithm, wrap="word", font=("Consolas", 10))
        self.algorithm_text.pack(fill="both", expand=True)

    def _build_patient_tab(self) -> None:
        wrapper = ttk.Frame(self.tab_patient)
        wrapper.pack(anchor="nw", fill="x", padx=10, pady=10)

        self.patient_vars = [
            tk.StringVar(value="Encendido (+1)"),
            tk.StringVar(value="Encendido (+1)"),
            tk.StringVar(value="Pulsador derecho (+1)"),
        ]
        options = [
            ("Pulsador izquierdo (x1):", self.patient_vars[0], ("Apagado (-1)", "Encendido (+1)")),
            ("Pulsador derecho (x2):", self.patient_vars[1], ("Apagado (-1)", "Encendido (+1)")),
            ("Elección paloma (x3):", self.patient_vars[2], ("Pulsador izquierdo (-1)", "Pulsador derecho (+1)")),
        ]
        for row, (label, var, vals) in enumerate(options):
            ttk.Label(wrapper, text=label, width=24).grid(row=row, column=0, sticky="w", pady=5)
            ttk.Combobox(wrapper, textvariable=var, values=vals, state="readonly", width=22).grid(
                row=row, column=1, sticky="w", pady=5
            )

        ttk.Button(wrapper, text="Evaluar elección de paloma", command=self._classify_patient).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(10, 8)
        )
        self.patient_result_var = tk.StringVar(value="Entrene primero el perceptrón.")
        ttk.Label(wrapper, textvariable=self.patient_result_var, style="Result.TLabel").grid(
            row=4, column=0, columnspan=3, sticky="w"
        )

    def _load_pattern_table(self) -> None:
        for item in self.pattern_tree.get_children():
            self.pattern_tree.delete(item)
        for index, pattern in enumerate(self.patterns):
            pred = "-"
            if self.result is not None:
                pred = str(self.result.final_predictions[index])
            self.pattern_tree.insert(
                "",
                "end",
                values=(
                    pattern.name,
                    pattern.inputs[0],
                    pattern.inputs[1],
                    pattern.inputs[2],
                    pattern.target,
                    pred,
                ),
            )

    def _show_algorithm_text(self) -> None:
        text = "ALGORITMO DE APRENDIZAJE DE LA PALOMA (PERCEPTRÓN SIMPLE)\n\n"
        text += "1. Cargar los 8 patrones bipolares X=(x1,x2,x3) y sus salidas deseadas T:\n"
        text += "   - x1: Estado del pulsador izquierdo (-1=apagado, +1=encendido)\n"
        text += "   - x2: Estado del pulsador derecho (-1=apagado, +1=encendido)\n"
        text += "   - x3: Elección de la paloma (-1=pulsador izq, +1=pulsador der)\n"
        text += "   - T: Salida deseada (+1=Éxito / consigue comida, -1=Fracaso / no consigue comida)\n\n"
        text += "2. Inicializar W=(w1,w2,w3) y bias b (Propuestos: W0=(1.2, 0.8, 2.1), b0=0.8).\n"
        text += "3. Para cada época, recorrer secuencialmente los 8 patrones:\n"
        text += "      a = w1*x1 + w2*x2 + w3*x3 + b\n"
        text += "      Y = +1 si a >= 0; de lo contrario Y = -1 (hardlims)\n"
        text += "      e = T - Y\n"
        text += "      W <- W + e*X\n"
        text += "      b <- b + e\n"
        text += "4. Al terminar la época, calcular el error global:\n"
        text += "      MSE = (1/N) * SUM(e(k)^2), con N=8\n"
        text += "5. Si MSE = 0, detener: la paloma ha aprendido la regla de manera óptima.\n"
        text += "6. En otro caso, iniciar una nueva época hasta max_epochs.\n\n"
        text += "Valores esperados tras entrenamiento:\n"
        text += "  - Convergencia en época 2 con MSE = [0.5, 0.0]\n"
        text += "  - Pesos finales W = (3.2, -1.2, 0.1), Bias final b = -1.2\n"
        self.algorithm_text.delete("1.0", "end")
        self.algorithm_text.insert("1.0", text)
        self.algorithm_text.configure(state="disabled")

    def _show_initial_graph(self) -> None:
        self.ax.clear()
        self.ax.set_title("Evolución del MSE global por época")
        self.ax.set_xlabel("Época")
        self.ax.set_ylabel("MSE global")
        self.ax.grid(True, alpha=0.3)
        self.ax.text(0.5, 0.5, "Entrene el perceptrón para generar la gráfica", ha="center", va="center", transform=self.ax.transAxes)
        self.canvas.draw_idle()

    def _read_parameters(self) -> tuple[list[float], float, int]:
        try:
            weights = [float(self.w1_var.get()), float(self.w2_var.get()), float(self.w3_var.get())]
            bias = float(self.bias_var.get())
            max_epochs = int(self.max_epochs_var.get())
        except ValueError as exc:
            raise ValueError("Pesos y bias deben ser numéricos; máximo de épocas debe ser entero.") from exc
        if max_epochs <= 0:
            raise ValueError("El máximo de épocas debe ser mayor que cero.")
        return weights, bias, max_epochs

    def _train(self) -> None:
        try:
            weights, bias, max_epochs = self._read_parameters()
        except ValueError as exc:
            messagebox.showerror("Parámetros inválidos", str(exc))
            return

        self.model = SimplePerceptron(weights=weights, bias=bias)
        self.result = train_perceptron(self.model, self.patterns, max_epochs=max_epochs)

        self._load_pattern_table()
        self._render_training_log()
        plot_mse(self.ax, self.result)
        self.figure.tight_layout()
        self.canvas.draw_idle()

        if self.result.converged:
            self.status_var.set(
                f"Convergencia en {self.result.epochs} época(s). "
                f"W={self._fmt_tuple(self.result.final_weights)}, b={self.result.final_bias:.4g}."
            )
        else:
            self.status_var.set(
                f"No convergió en {self.result.epochs} época(s). "
                f"Último MSE={self.result.metrics[-1].mse:.4g}."
            )

        self.patient_result_var.set("Perceptrón entrenado. Puede evaluar una nueva elección de la paloma.")
        self.notebook.select(self.tab_training)

    def _render_training_log(self) -> None:
        assert self.result is not None
        self.training_text.delete("1.0", "end")

        self.training_text.insert("end", "ENTRENAMIENTO PASO A PASO - CASO 1: PALOMA\n")
        self.training_text.insert("end", "=" * 96 + "\n")

        for metric in self.result.metrics:
            self.training_text.insert("end", f"\nÉPOCA {metric.epoch}\n")
            self.training_text.insert("end", "-" * 96 + "\n")
            epoch_steps = [step for step in self.result.steps if step.epoch == metric.epoch]
            for step in epoch_steps:
                self.training_text.insert(
                    "end",
                    f"{step.pattern_name}: X={step.inputs}, T={step.target}\n"
                    f"  W antes={self._fmt_tuple(step.weights_before)}, b antes={step.bias_before:.4g}\n"
                    f"  a = W·X + b = {step.net:.4g}\n"
                    f"  Y = hardlims(a) = {step.output}\n"
                    f"  e = T - Y = {step.error}; e² = {step.squared_error}\n"
                    f"  W después={self._fmt_tuple(step.weights_after)}, b después={step.bias_after:.4g}"
                    + ("  <- actualización\n" if step.updated else "  <- sin cambio efectivo\n")
                )

            self.training_text.insert(
                "end",
                f"\nResumen época {metric.epoch}: SSE={metric.sse:.4g}; "
                f"MSE=SSE/8={metric.mse:.4g}; "
                f"patrones mal clasificados durante la pasada={metric.misclassified_during_pass}; "
                f"actualizaciones={metric.updates}.\n"
                f"W fin={self._fmt_tuple(metric.final_weights)}, b fin={metric.final_bias:.4g}.\n"
            )

        self.training_text.insert("end", "\n" + "=" * 96 + "\n")
        self.training_text.insert(
            "end",
            f"Resultado final: convergió={self.result.converged}; épocas={self.result.epochs}; "
            f"W={self._fmt_tuple(self.result.final_weights)}; b={self.result.final_bias:.4g}.\n",
        )

    def _classify_patient(self) -> None:
        if self.model is None or self.result is None or not self.result.converged:
            messagebox.showinfo("Entrenamiento requerido", "Entrene primero el perceptrón hasta convergencia.")
            return

        v1 = 1 if "Encendido" in self.patient_vars[0].get() else -1
        v2 = 1 if "Encendido" in self.patient_vars[1].get() else -1
        v3 = 1 if "derecho" in self.patient_vars[2].get() else -1
        inputs = (v1, v2, v3)

        net = self.model.net_input(inputs)
        output = self.model.hardlims(net)
        resultado = "ÉXITO (+1): La paloma consigue comida" if output == 1 else "FRACASO (-1): No consigue comida"
        self.patient_result_var.set(
            f"X={inputs} | a={net:.4g} | Y={output} -> {resultado}"
        )

    def _reset(self) -> None:
        self.w1_var.set(str(DEFAULT_WEIGHTS[0]))
        self.w2_var.set(str(DEFAULT_WEIGHTS[1]))
        self.w3_var.set(str(DEFAULT_WEIGHTS[2]))
        self.bias_var.set(str(DEFAULT_BIAS))
        self.max_epochs_var.set(str(DEFAULT_MAX_EPOCHS))
        self.result = None
        self.model = None
        self.status_var.set("Listo para entrenar.")
        self.patient_result_var.set("Entrene primero el perceptrón.")
        self.training_text.delete("1.0", "end")
        self._load_pattern_table()
        self._show_initial_graph()

    @staticmethod
    def _fmt_tuple(values: tuple[float, ...] | list[float]) -> str:
        return "(" + ", ".join(f"{value:.4g}" for value in values) + ")"
