# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os


class GradientBoostingModel:
    def __init__(self, csv_path, model_id, target_column="kategorie", drop_columns=None, test_size=0.2):
        self.csv_path = csv_path
        self.model_id = model_id
        self.target_column = target_column
        self.drop_columns = drop_columns or ["cu_id"]
        self.test_size = test_size
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model = HistGradientBoostingClassifier(random_state=42)

    def load_data(self):
        self.df = pd.read_csv(self.csv_path, sep=";")
        print(f"[{self.model_id}] CSV geladen mit {self.df.shape[0]} Zeilen und {self.df.shape[1]} Spalten.")

        # Zielvariable in binär umwandeln
        self.df[self.target_column] = (self.df[self.target_column] == 1).astype(int)
        print(f"[{self.model_id}] 'kategorie' in binäre Zielvariable umgewandelt.")

        print(f"\n[{self.model_id}] Klassenverteilung:")
        print(self.df[self.target_column].value_counts())

    def prepare_data(self):
        X = self.df.drop(columns=self.drop_columns + [self.target_column], errors="ignore")
        y = self.df[self.target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print(f"[{self.model_id}] Trainingsdaten: {self.X_train.shape}, Testdaten: {self.X_test.shape}")

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
        print(f"[{self.model_id}] Modelltraining abgeschlossen (Gradient Boosting).")

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)

        report = classification_report(
            self.y_test, y_pred, labels=[0, 1], target_names=["Schlecht", "Gut"]
        )
        print(f"\n[{self.model_id}] Klassifikationsbericht:")
        print(report)

        cm = confusion_matrix(self.y_test, y_pred, labels=[0, 1])
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                    xticklabels=["Schlecht", "Gut"],
                    yticklabels=["Schlecht", "Gut"])
        plt.title(f"Konfusionsmatrix Modell {self.model_id}")
        plt.xlabel("Vorhergesagt")
        plt.ylabel("Tatsächlich")
        plt.tight_layout()

        os.makedirs("Konfusionsmatrix", exist_ok=True)
        os.makedirs("Klassifikationsberichte", exist_ok=True)

        matrix_path = os.path.join("Konfusionsmatrix", f"konfusionsmatrix_{self.model_id}.png")
        report_path = os.path.join("Klassifikationsberichte", f"bericht_{self.model_id}.txt")

        plt.savefig(matrix_path)
        plt.close()
        print(f"[{self.model_id}] Konfusionsmatrix gespeichert unter: {matrix_path}")

        with open(report_path, "w") as f:
            f.write(report)
        print(f"[{self.model_id}] Klassifikationsbericht gespeichert unter: {report_path}")

    def run_all(self):
        self.load_data()
        self.prepare_data()
        self.train_model()
        self.evaluate_model()


def run_multiple_gradient_boosting_models(csv_paths):
    for idx, path in enumerate(csv_paths, start=1):
        print(f"\n===== Verarbeitung von Datei {idx}: {path} =====")
        model = GradientBoostingModel(csv_path=path, model_id=idx)
        model.run_all()
