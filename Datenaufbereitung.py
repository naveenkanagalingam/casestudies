# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 14:40:19 2025

@author: navee
"""

# -*- coding: utf-8 -*-
"""
OOP-Converter: Rohdaten (DOEs.xlsx) in analysierbares Format (1 Zeile = 1 xy-Punkt)
Autor: ssecc (überarbeitet durch ChatGPT)
"""

import pandas as pd
import os

class DOEConverter:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.xl_file = pd.ExcelFile(input_path, engine="openpyxl")
        self.sheet_names = self.xl_file.sheet_names

    def process_all_sheets(self):
        writer = pd.ExcelWriter(self.output_path, engine="openpyxl")

        for sheet in self.sheet_names:
            print(f"Verarbeite Sheet: {sheet}")
            try:
                df = pd.read_excel(self.xl_file, sheet_name=sheet)
                df_converted = self._process_single_sheet(df)
                if not df_converted.empty:
                    df_converted.to_excel(writer, sheet_name=sheet[:31], index=False)
            except Exception as e:
                print(f"Fehler beim Verarbeiten von Sheet '{sheet}': {e}")

        writer.close()
        print(f"\nAlle Daten gespeichert unter:\n{self.output_path}")

    def _process_single_sheet(self, df):
        alle_daten_sheet = []

        for index, row in df.iterrows():
            try:
                cu_id = row['cu_id']
                raw_cu_x = row['cu_x']
                raw_cu_y = row['cu_y']
                kategorie = row.get('cu_machine_parameter_values_id', None)

                if pd.notna(raw_cu_x) and pd.notna(raw_cu_y):
                    liste_cu_x = [float(x.strip()) for x in raw_cu_x.strip('{}').split(',')]
                    liste_cu_y = [float(y.strip()) for y in raw_cu_y.strip('{}').split(',')]

                    if len(liste_cu_x) == len(liste_cu_y):
                        df_temp = pd.DataFrame({
                            'cu_id': cu_id,
                            'x_achse': liste_cu_x,
                            'y_achse': liste_cu_y,
                            'Kategorie': kategorie
                        })
                        alle_daten_sheet.append(df_temp)
                    else:
                        print(f"Länge nicht gleich bei cu_id {cu_id} (x: {len(liste_cu_x)}, y: {len(liste_cu_y)})")

            except Exception as e:
                print(f"Fehler in Zeile {index}: {e}")

        if alle_daten_sheet:
            return pd.concat(alle_daten_sheet, ignore_index=True)
        else:
            return pd.DataFrame()
