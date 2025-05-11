# -*- coding: utf-8 -*-
"""
Created on Fri May  9 16:00:34 2025

@author: ssecc
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class FeatureFormparameterAnalyzer:
    def __init__(self, df, kategorien_filter=[1]):
        self.df = df[df["Kategorie"].isin(kategorien_filter)].copy()
        self.kategorien_filter = kategorien_filter
        self.x_mean = None
        self.y_mean = None
        self.slopes = None
        self.slope_means = None
        self.quartile_indices = None

    def calculate_mean_curve(self):
        df_sorted = self.df.sort_values(by=["cu_id", "x_achse"])
        pivot = df_sorted.pivot(index="x_achse", columns="cu_id", values="y_achse")
        self.x_mean = pivot.index.to_numpy()
        self.y_mean = pivot.mean(axis=1).to_numpy()

    def calculate_steigungen(self):
        dy = np.gradient(self.y_mean)
        dx = np.gradient(self.x_mean)
        self.slopes = dy / dx

        # Max/Min Steigung
        self.slope_max = np.max(self.slopes)
        self.slope_min = np.min(self.slopes)

        self.slope_max_index = np.argmax(self.slopes)
        self.slope_min_index = np.argmin(self.slopes)
        self.slope_max_x = self.x_mean[self.slope_max_index]
        self.slope_max_y = self.y_mean[self.slope_max_index]
        self.slope_min_x = self.x_mean[self.slope_min_index]
        self.slope_min_y = self.y_mean[self.slope_min_index]

        # Quartile
        self.quartile_indices = np.array_split(np.arange(len(self.slopes)), 4)
        self.slope_means = [np.mean(self.slopes[idx]) for idx in self.quartile_indices]

    def plot_curve_marked(self):
        quartile_labels = ["slope_early", "slope_mid1", "slope_mid2", "slope_late"]
        quartile_colors = ["#b3e2cd", "#fdcdac", "#cbd5e8", "#f4cae4"]

        plt.figure(figsize=(10, 5))
        plt.plot(self.x_mean, self.y_mean, color="blue", label="Durchschnittskurve")

        for i, idx in enumerate(self.quartile_indices):
            plt.axvspan(self.x_mean[idx[0]], self.x_mean[idx[-1]], alpha=0.15, color=quartile_colors[i], label=quartile_labels[i])
            xm = self.x_mean[idx[len(idx)//2]]
            ym = self.y_mean[idx[len(idx)//2]]
            plt.plot(xm, ym, "o", color=quartile_colors[i], markersize=6)
            plt.text(xm, ym + 10, f"{quartile_labels[i]}\n{self.slope_means[i]:.1f}", ha="center", fontsize=9, color="black")

        # Markierung von slope_max
        plt.plot(self.slope_max_x, self.slope_max_y, "o", color="red", label="slope_max")
        plt.text(self.slope_max_x, self.slope_max_y + 15, f"max: {self.slope_max:.1f}", ha="center", fontsize=9, color="red")

        # Markierung von slope_min
        plt.plot(self.slope_min_x, self.slope_min_y, "o", color="purple", label="slope_min")
        plt.text(self.slope_min_x, self.slope_min_y - 25, f"min: {self.slope_min:.1f}", ha="center", fontsize=9, color="purple")

        # x-Achse einteilen
        x_start = self.x_mean[0]
        x_end = self.x_mean[-1]
        x_ticks = np.linspace(x_start, x_end, 5)
        plt.xticks(x_ticks)

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Durchschnittskurve mit Steigungen und Extrempunkten")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# --- Hauptprogramm ---

# Excel-Datei laden
excel_path = "C:/Users/ssecc/OneDrive/Dokumente/ZHAW/ZHAW SEMESTER 6/POM4/DOE_Gesamtauswertung_verarbeitet.xlsx"
df = pd.read_excel(excel_path, sheet_name="DOE1")

# Kategorie(n) definieren
kategorien_filter = [1]

# Analyzer ausführen
analyzer = FeatureFormparameterAnalyzer(df, kategorien_filter)
analyzer.calculate_mean_curve()
analyzer.calculate_steigungen()

# Steigungen ausgeben
print("\n--- Steigungsmerkmale ---")
print(f"slope_max: {analyzer.slope_max:.2f} bei x = {analyzer.slope_max_x:.2f}")
print(f"slope_min: {analyzer.slope_min:.2f} bei x = {analyzer.slope_min_x:.2f}")
for i, s in enumerate(analyzer.slope_means, start=1):
    print(f"Abschnitt {i} (slope_{['early', 'mid1', 'mid2', 'late'][i-1]}): {s:.2f}")

# Plot anzeigen
analyzer.plot_curve_marked()
