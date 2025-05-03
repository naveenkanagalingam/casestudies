# -*- coding: utf-8 -*-
"""
Created on Sat May  3 11:21:35 2025

@author: ssecc
"""


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class MLModel:
    def __init__(self, csv_path, target_column="kategorie", drop_columns=None, test_size=0.2):
        self.csv_path = csv_path
        self.target_column = target_column
        self.drop_columns = drop_columns or ["cu_id"]
        self.test_size = test_size
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def load_data(self):
        self.df = pd.read_csv(self.csv_path, sep=";")
        print(f"✅ CSV geladen mit {self.df.shape[0]} Zeilen und {self.df.shape[1]} Spalten.")

        # Zielvariable in binär umwandeln: 1 = perfekt, 0 = schmutz
        self.df[self.target_column] = (self.df[self.target_column] == 1).astype(int)
        print("🔁 'kategorie' wurde in binäre Zielvariable umgewandelt (1 = perfekt, 0 = schmutz).")

        # Klassenverteilung anzeigen
        print("\n📊 Klassenverteilung (nach Umwandlung):")
        print(self.df[self.target_column].value_counts())

    def prepare_data(self):
        X = self.df.drop(columns=self.drop_columns + [self.target_column], errors="ignore")
        y = self.df[self.target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print(f"\n📂 Trainingsdaten: {self.X_train.shape}, Testdaten: {self.X_test.shape}")

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
        print("✅ Modelltraining abgeschlossen.")

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)

        print("\n🔍 Klassifikationsbericht:")
        print(classification_report(
            self.y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Schlecht", "Gut"]
        ))

        cm = confusion_matrix(self.y_test, y_pred, labels=[0, 1])
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Schlecht", "Gut"], yticklabels=["Schlecht", "Gut"])
        plt.title("Konfusionsmatrix")
        plt.xlabel("Vorhergesagt")
        plt.ylabel("Tatsächlich")
        plt.tight_layout()
        plt.show()

    def run_all(self):
        self.load_data()
        self.prepare_data()
        self.train_model()
        self.evaluate_model()

