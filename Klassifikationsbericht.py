import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path

class ClassificationReportParser:
    def __init__(self, filepath: Path, model_name: str):
        self.filepath = filepath
        self.model_name = model_name
        self.df = self._parse_file()

    def _parse_file(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        data = []
        for line in lines:
            if re.match(r"\s*(nicht akzeptabel|akzeptabel)", line):
                parts = re.split(r"\s{2,}", line.strip())
                if len(parts) >= 4:
                    klasse = parts[0]
                    precision = float(parts[1])
                    recall = float(parts[2])
                    f1 = float(parts[3])
                    data.append((klasse, precision, recall, f1))
        df = pd.DataFrame(data, columns=["Kategorie", "Precision", "Recall", "F1-Score"])
        df["Modell"] = self.model_name
        return df

class EvaluationPlotter:
    def __init__(self):
        self.dataframes = []
        self.model_names = []

    def add_model_report(self, filepath: str, model_name: str):
        parser = ClassificationReportParser(Path(filepath), model_name)
        self.dataframes.append(parser.df)
        self.model_names.append(model_name.replace(" ", "_"))

    def plot(self, save_path: str = None):
        if not self.dataframes:
            raise ValueError("Keine Daten vorhanden.")
        df_all = pd.concat(self.dataframes, ignore_index=True)
        df_long = df_all.melt(id_vars=["Kategorie", "Modell"], var_name="Metrik", value_name="Wert")

        g = sns.catplot(
            data=df_long,
            kind="bar",
            x="Metrik",
            y="Wert",
            hue="Modell",
            col="Kategorie",
            palette="Set2",
            height=5,
            aspect=1
        )

        for ax in g.axes.flatten():
            for container in ax.containers:
                ax.bar_label(container, fmt="%.2f", label_type="edge", padding=2)

        g.set_titles("Kategorie: {col_name}")
        g.set(ylim=(0, 1.05))
        g.set_axis_labels("Metrik", "Wert")
        g.fig.suptitle("Klassifikationsmetriken pro Modell und Kategorie", y=1.05)
        g.tight_layout()

        # Plot speichern, wenn Pfad angegeben oder automatisch erstellen
        if save_path is None:
            filename = "_".join(self.model_names) + ".png"
            save_path = Path("Klassifikationsberichte") / filename
        else:
            save_path = Path(save_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        g.fig.savefig(save_path, dpi=300)
        print(f"Plot gespeichert unter: {save_path}")

        plt.close()

# # DOE1
# plotter = EvaluationPlotter()
# plotter.add_model_report("Klassifikationsberichte/GBM_bericht_1.txt", "Gradient Boosting")
# plotter.add_model_report("Klassifikationsberichte/LR_bericht_1.txt", "Logistische Regression")
# plotter.add_model_report("Klassifikationsberichte/RFC_bericht_1.txt", "Random Forest")
# plotter.plot() 

# # DOE2
# plotter = EvaluationPlotter()
# plotter.add_model_report("Klassifikationsberichte/GBM_bericht_2.txt", "Gradient Boosting")
# plotter.add_model_report("Klassifikationsberichte/LR_bericht_2.txt", "Logistische Regression")
# plotter.add_model_report("Klassifikationsberichte\RFC_bericht_2.txt", "Random Forest")
# plotter.plot()

# DOE3
plotter = EvaluationPlotter()
plotter.add_model_report("Klassifikationsberichte/GBM_bericht_3.txt", "Gradient Boosting")
plotter.add_model_report("Klassifikationsberichte/LR_bericht_3.txt", "Logistische Regression")
plotter.add_model_report("Klassifikationsberichte/RFC_bericht_3.txt", "Random Forest")
plotter.plot()