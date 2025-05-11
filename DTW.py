# -*- coding: utf-8 -*-
"""
Created on [Datum hier eintragen]
@author: ssecc
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dtaidistance import dtw

class DTWHeatmapPlotter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.sheets = {}

    def lade_sheets(self):
        self.sheets = pd.read_excel(self.file_path, sheet_name=None, engine="openpyxl")

    def berechne_dtw_distanzen(self, df):
        df = df.copy()
        df = df.sort_values(by=['cu_id', 'x_achse'])
        df['Kategorie'] = df['Kategorie'].astype(int)

        df_ref = df[df['Kategorie'] == 1]
        ref_mean = df_ref.groupby('x_achse')['y_achse'].mean().values

        distanzen = []
        for cu_id, gruppe in df.groupby('cu_id'):
            y_seq = gruppe.sort_values('x_achse')['y_achse'].values
            dist = dtw.distance(ref_mean, y_seq)
            norm_dist = dist / len(y_seq)  # Normalisierung
            kat = gruppe['Kategorie'].iloc[0]
            distanzen.append({'cu_id': cu_id, 'dtw_distanz': norm_dist, 'Kategorie': kat})

        return pd.DataFrame(distanzen)

    def plot_dtw_verteilungen(self, df_distanzen, sheetname):
        os.makedirs("Plots_DTW_Verteilungen", exist_ok=True)

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_distanzen, x='Kategorie', y='dtw_distanz', palette='tab10')
        plt.title(f"Verteilung der DTW-Distanzen zur Referenz (Klasse 1) – {sheetname}")
        plt.xlabel("Kategorie")
        plt.ylabel("Normalisierte DTW-Distanz")
        plt.tight_layout()
        plt.savefig(f"Plots_DTW_Verteilungen/DTW_Verteilung_{sheetname}.png", dpi=300)
        plt.close()

    def plot_heatmap(self):
        os.makedirs("Plots_Heatmaps", exist_ok=True)

        for sheetname, df in self.sheets.items():
            df = df.copy()
            df = df.sort_values(by=['Kategorie', 'cu_id', 'x_achse'])

            # DTW-Distanzverteilung zur Referenzkurve (Klasse 1)
            df_distanzen = self.berechne_dtw_distanzen(df)
            self.plot_dtw_verteilungen(df_distanzen, sheetname)

            # Erstelle Durchschnittskurven pro Kategorie
            df_means = df.groupby(['Kategorie', 'x_achse'])['y_achse'].mean().reset_index()
            gruppen = [g['y_achse'].values for _, g in df_means.groupby('Kategorie')]
            labels = list(df_means['Kategorie'].unique())

            # Berechne DTW-Distanzen zwischen den Durchschnittskurven
            n = len(gruppen)
            matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    d = dtw.distance(gruppen[i], gruppen[j])
                    matrix[i, j] = d
                    matrix[j, i] = d

            plt.figure(figsize=(8, 6))
            sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap="viridis", square=True)
            plt.title(f"DTW-Distanzmatrix – Durchschnittskurven pro Kategorie – {sheetname}")
            plt.xlabel("Kategorie")
            plt.ylabel("Kategorie")
            plt.tight_layout()
            plt.savefig(f"Plots_Heatmaps/DTW_Heatmap_{sheetname}.png", dpi=300)
            plt.close()

    def ausführen(self):
        self.lade_sheets()
        self.plot_heatmap()

if __name__ == "__main__":
    pfad = "C:/Users/navee/OneDrive/Desktop/CaseStudies/xlsx_files/DOEs_aufbereitet.xlsx"
    plotter = DTWHeatmapPlotter(pfad)
    plotter.ausführen()
