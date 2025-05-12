import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ModelMetricsPlotter:
    def __init__(self):
        self.metrics_data = []

    def add_model_metrics(self, model_name: str, accuracy: float, f1_akzeptabel: float, macro_f1: float, weighted_f1: float):
        self.metrics_data.append({
            "Modell": model_name,
            "Accuracy": accuracy,
            "F1_akzeptabel": f1_akzeptabel,
            "Macro_F1": macro_f1,
            "Weighted_F1": weighted_f1
        })

    def plot(self, save_path: str = None):
        if not self.metrics_data:
            raise ValueError("Keine Modellmetriken vorhanden.")

        # In DataFrame umwandeln
        df_mean = pd.DataFrame(self.metrics_data)
        df_mean = df_mean.set_index("Modell").T.reset_index().rename(columns={"index": "Metrik"})
        df_long = df_mean.melt(id_vars="Metrik", var_name="Modell", value_name="Wert")

        # Plot erzeugen
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df_long, x="Metrik", y="Wert", hue="Modell", palette="Set2")

        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", label_type="edge", padding=2)

        plt.ylim(0, 1.05)
        plt.title("Durchschnittliche Modellmetriken im Vergleich")
        plt.ylabel("Wert")

        # Legende ausserhalb des Plots
        plt.legend(title="Modell", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

        plt.tight_layout()

        # Speicherpfad definieren, falls nicht angegeben
        if save_path is None:
            save_path = Path("Klassifikationsberichte") / "durchschnittliche_modellmetriken_im_vergleich.png"
        else:
            save_path = Path(save_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot gespeichert unter: {save_path}")

        plt.show()
        plt.close()

# Beispielverwendung
plotter = ModelMetricsPlotter()
plotter.add_model_metrics("Random Forest", 0.94, 0.51, 0.74, 0.92)
plotter.add_model_metrics("Logistische Regression", 0.96, 0.75, 0.86, 0.95)
plotter.add_model_metrics("Gradient Boosting", 0.93, 0.76, 0.86, 0.94)
plotter.plot()
