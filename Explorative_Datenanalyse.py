# -*- coding: utf-8 -*-
"""
Created on Sat May  3 11:36:44 2025

@author: ssecc
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr

class CurveAnalyzer:
    def __init__(self, original_path, kategorien_auswahl):
        self.original_path = original_path
        self.kategorien_auswahl = kategorien_auswahl
        self.output_path = f"{original_path[:-5]}_alle_sheets_neu.xlsx"
        self.sheet_dfs = {}  # Dict für bearbeitete DataFrames pro Sheet
        self.palette = sns.color_palette("hls", len(kategorien_auswahl))
        self.farben = dict(zip(kategorien_auswahl, self.palette))
        self.farben[1] = 'gray'  # Kategorie 1 soll grau dargestellt werden

    def lade_alle_sheets(self):
        self.original_sheets = pd.read_excel(self.original_path, sheet_name=None, engine="openpyxl")

    def finde_ausreisser_in_sheet(self, df_analyse, korrelation_threshold=0.98):
        df_ref = df_analyse[df_analyse['Kategorie'] == 1]
        df_test = df_analyse[df_analyse['Kategorie'] != 1]

        if df_ref.empty:
            return pd.DataFrame(), [], []

        ref_mean = df_ref.groupby('x_achse')['y_achse'].mean()

        ausreisser_liste = []
        cu_ids_mit_abweichung = []
        cu_ids_ähnlich = []

        for cu_id, gruppe in df_test.groupby('cu_id'):
            y_interp = ref_mean.reindex(gruppe['x_achse'], method='nearest')
            y_target = gruppe['y_achse'].values
            y_ref = y_interp.values

            if len(y_target) != len(y_ref):
                continue

            corr, _ = pearsonr(y_target, y_ref)

            if corr >= korrelation_threshold:
                cu_ids_ähnlich.append(cu_id)
            else:
                ausreisser_liste.append(gruppe)
                cu_ids_mit_abweichung.append(cu_id)

        df_ausreisser = pd.concat(ausreisser_liste, ignore_index=True) if ausreisser_liste else pd.DataFrame()
        return df_ausreisser, cu_ids_mit_abweichung, cu_ids_ähnlich

    def verarbeite_sheets(self):
        for sheetname, df_full in self.original_sheets.items():
            print(f"\nBearbeite Sheet: {sheetname}")
            df_analyse = df_full[df_full['Kategorie'].isin(self.kategorien_auswahl)].copy()
            df_analyse = df_analyse.sort_values(by=['cu_id', 'x_achse'])

            # 📌 Vor der Anpassung: Für Plot zwischenspeichern
            df_plot_original = df_full.copy()

            df_ausreisser, cu_ids_mit_ausreisser, cu_ids_ähnlich = self.finde_ausreisser_in_sheet(df_analyse)

            df_full.loc[
                (df_full['cu_id'].isin(cu_ids_ähnlich)) & (df_full['Kategorie'] != 1),
                'Kategorie'
            ] = 1

            self.sheet_dfs[sheetname] = df_full

            print(f"Ausreißer in {sheetname} erkannt: {len(cu_ids_mit_ausreisser)} cu_ids mit Abweichungen.")
            if cu_ids_ähnlich:
                print(f"Kategorie 1 gesetzt für ähnliche cu_ids: {sorted(cu_ids_ähnlich)}")

            # 📌 Plot mit ursprünglichen Kategorien (vor Anpassung)
            self.plot_kurven_mit_ausreissern(df_plot_original, df_ausreisser, cu_ids_mit_ausreisser, sheetname)

    def speichere_alle_sheets(self):
        with pd.ExcelWriter(self.output_path, engine="openpyxl") as writer:
            for sheetname, df in self.sheet_dfs.items():
                df.to_excel(writer, sheet_name=sheetname, index=False)
        print(f"\nAlle bearbeiteten Sheets gespeichert unter:\n{self.output_path}")

    def plot_kurven_mit_ausreissern(self, df_full, df_ausreisser, cu_ids_mit_ausreisser, sheetname):
        os.makedirs("Plots", exist_ok=True)

        plt.figure(figsize=(10, 6))

        df_plot = df_full[df_full['Kategorie'].isin(self.kategorien_auswahl)]

        for cu_id, gruppe in df_plot.groupby('cu_id'):
            kat = gruppe['Kategorie'].iloc[0]
            farbe = self.farben.get(kat, 'gray')
            label = 'akzeptabel' if kat == 1 else 'nicht akzeptabel'
            if label not in plt.gca().get_legend_handles_labels()[1]:
                plt.plot(gruppe['x_achse'], gruppe['y_achse'], color=farbe, alpha=0.7, linewidth=0.8, label=label)
            else:
                plt.plot(gruppe['x_achse'], gruppe['y_achse'], color=farbe, alpha=0.7, linewidth=0.8)

        if not df_ausreisser.empty:
            plt.scatter(df_ausreisser['x_achse'], df_ausreisser['y_achse'], color='red', s=10, label='Ausreißer')

        titel = f"CU-Kurven mit Ausreißern (rot markiert)"
        if cu_ids_mit_ausreisser:
            titel += f"\nCU-IDs mit Ausreißern: {', '.join(map(str, cu_ids_mit_ausreisser))}"

        plt.title(titel)
        plt.xlabel("Zeit in s")
        plt.ylabel("Druck in bar")
        plt.legend()
        plt.tight_layout()
        kategorien_str = "_".join(map(str, self.kategorien_auswahl))
        plot_path = f"Plots/{sheetname}_({kategorien_str}).png"
        plt.savefig(plot_path, dpi=300)
        plt.show()
        plt.close()

    def ausführen(self):
        self.lade_alle_sheets()
        self.verarbeite_sheets()
        self.speichere_alle_sheets()

if __name__ == "__main__":
    pfad = "C:/Users/navee/OneDrive/Desktop/CaseStudies/xlsx_files/DOEs_aufbereitet.xlsx"
    eingabe = input("Gib die Kategorien-Auswahl ein (z. B. 1,2 oder 1 2): ").replace(",", " ")
    kategorien_auswahl = [int(k) for k in eingabe.split() if k.strip().isdigit()]

    analyzer = CurveAnalyzer(pfad, kategorien_auswahl)
    analyzer.ausführen()
