# -*- coding: utf-8 -*-
"""
Created on Sun May 11 17:22:26 2025
@author: navee
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Parameter
excel_path = "C:/Users/navee/OneDrive/Desktop/CaseStudies/xlsx_files/DOEs_aufbereitet_alle_sheets_neu.xlsx"
prominenz_schwelle = 0.5  # Mindesthöhe für Peaks/Valleys

# Excel einlesen
excel = pd.read_excel(excel_path, sheet_name=None)

for sheet_name, df in excel.items():
    print(f"\nSheet: {sheet_name}")

    df_cu1 = df[df['cu_id'] == 1].copy()
    if df_cu1.empty:
        print("cu_id 1 nicht vorhanden.")
        continue

    df_cu1 = df_cu1.sort_values('x_achse')
    x = df_cu1['x_achse'].values
    y = df_cu1['y_achse'].values

    # Lokale Maxima und Minima direkt aus der Rohkurve
    max_idx, _ = find_peaks(y, prominence=prominenz_schwelle)
    min_idx, _ = find_peaks(-y, prominence=prominenz_schwelle)

    print(f"Maxima: {len(max_idx)} – Minima: {len(min_idx)}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label="cu_id 1", color='orange', linewidth=2)
    plt.scatter(x[max_idx], y[max_idx], color='red', label='Maxima')
    plt.scatter(x[min_idx], y[min_idx], color='blue', label='Minima')

    plt.title(f"Referenzkurve (akzeptabel) – {sheet_name}")
    plt.xlabel("Zeit in s")
    plt.ylabel("Druck in bar")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()