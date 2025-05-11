#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 17:47:10 2025

@author: huppimarc
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_doe_data(excel_file_path, sheet_name):
    """Lädt den gewünschten DOE-Datensatz aus dem Excel-Sheet."""
    return pd.read_excel(excel_file_path, sheet_name=sheet_name)

def extract_features(df, category_selection):
    feature_list = []
    categories = df['Kategorie'].unique() if category_selection == 'all' else [int(category_selection)]

    for category in categories:
        df_cat = df[df['Kategorie'] == category]
        for cu_id in df_cat['cu_id'].unique():
            curve = df_cat[df_cat['cu_id'] == cu_id]['y_achse'].values
            signal_energy = np.sum(np.square(curve))  # Gesamte Energie = Summe der quadrierten y-Werte
            total_length = len(curve)
            feature_list.append({
                "cu_id": cu_id,
                "Kategorie": category,
                "global_max_y": np.max(curve),
                "global_min_y": np.min(curve),
                "mean_curve_y": np.mean(curve),
                "median_curve_y": np.median(curve),
                "sd_curve_y": np.std(curve),
                "Diff_gmax_gmin": np.ptp(curve),
                "signal_energy": signal_energy,
                "total_curve_length": total_length
            })
    return pd.DataFrame(feature_list)

def plot_form_analysis(df, category_selection, features_df):
    category_selection = int(category_selection)
    df_cat = df[df['Kategorie'] == category_selection]
    curves = [df_cat[df_cat['cu_id'] == cu_id]['y_achse'].values for cu_id in df_cat['cu_id'].unique()]
    min_len = min(map(len, curves))
    curves_aligned = [c[:min_len] for c in curves]
    avg_curve = np.mean(curves_aligned, axis=0)

    feature_means = features_df[features_df['Kategorie'] == category_selection].mean()

    plt.figure(figsize=(12, 7))
    x_axis = np.arange(len(avg_curve))
    plt.plot(x_axis, avg_curve, label='Durchschnittskurve', color='royalblue', linewidth=2)

    # Feature-Werte
    g_max = feature_means['global_max_y']
    g_min = feature_means['global_min_y']
    diff = feature_means['Diff_gmax_gmin']
    idx_max = np.argmax(avg_curve)
    idx_min = np.argmin(avg_curve)

    # Punkte markieren (ohne Beschriftung im Plot)
    plt.scatter(idx_max, avg_curve[idx_max], color='red', s=100, marker='o')
    plt.scatter(idx_min, avg_curve[idx_min], color='purple', s=100, marker='o')

    # Doppelpfeil visualisieren (nur im Plot, kein Text)
    plt.annotate(
        '', xy=(idx_max, avg_curve[idx_max]), xytext=(idx_max, avg_curve[idx_min]),
        arrowprops=dict(arrowstyle='<->', color='black', linewidth=2)
    )

    # Dummy-Plots für Legende mit Werten
    plt.plot([], [], color='red', marker='o', linestyle='None', markersize=8, label=f'Maximum: {g_max:.2f}')
    plt.plot([], [], color='purple', marker='o', linestyle='None', markersize=8, label=f'Minimum: {g_min:.2f}')
    plt.plot([], [], color='black', linestyle='--', linewidth=2, label=f'Differenz Δy: {diff:.2f}')

    # Plot-Layout
    plt.xlabel('Zeit / Index', fontsize=13)
    plt.ylabel('Druck / Wert', fontsize=13)
    plt.title(f'Formanalyse Kategorie {category_selection}', fontsize=15)
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    plt.legend(loc='best', fontsize=11, frameon=True, fancybox=True)
    plt.tight_layout()
    plt.show()

def plot_trend_symmetry(df, category_selection, features_df):
    category_selection = int(category_selection)
    df_cat = df[df['Kategorie'] == category_selection]
    curves = [df_cat[df_cat['cu_id'] == cu_id]['y_achse'].values for cu_id in df_cat['cu_id'].unique()]
    min_len = min(map(len, curves))
    curves_aligned = [c[:min_len] for c in curves]
    avg_curve = np.mean(curves_aligned, axis=0)

    feature_means = features_df[features_df['Kategorie'] == category_selection].mean()
    mean_y = feature_means['mean_curve_y']
    median_y = feature_means['median_curve_y']

    plt.figure(figsize=(12, 7))
    x_axis = np.arange(len(avg_curve))

    # Durchschnittskurve
    plt.plot(x_axis, avg_curve, label='Durchschnittskurve', color='royalblue', linewidth=2)

    # Mittelwert- und Median-Linien (ohne Text direkt im Plot)
    plt.axhline(mean_y, color='green', linestyle='--', linewidth=2)
    plt.axhline(median_y, color='orange', linestyle='-.', linewidth=2)

    # Dummy-Plots für Legende mit den Werten
    plt.plot([], [], color='green', linestyle='--', linewidth=2, label=f'Mittelwert: {mean_y:.2f}')
    plt.plot([], [], color='orange', linestyle='-.', linewidth=2, label=f'Median: {median_y:.2f}')

    # Layout-Optimierung
    plt.xlabel('Zeit / Index', fontsize=13)
    plt.ylabel('Druck / Wert', fontsize=13)
    plt.title(f'Trend- & Symmetrieanalyse Kategorie {category_selection}', fontsize=15)
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    plt.legend(loc='best', fontsize=11, frameon=True, fancybox=True)
    plt.tight_layout()
    plt.show()

    
def plot_variability(df, category_selection, features_df):
    category_selection = int(category_selection)
    df_cat = df[df['Kategorie'] == category_selection]
    curves = [df_cat[df_cat['cu_id'] == cu_id]['y_achse'].values for cu_id in df_cat['cu_id'].unique()]
    min_len = min(map(len, curves))
    curves_aligned = [c[:min_len] for c in curves]
    avg_curve = np.mean(curves_aligned, axis=0)
    std_curve = np.std(curves_aligned, axis=0)

    feature_means = features_df[features_df['Kategorie'] == category_selection].mean()
    std_mean = feature_means['sd_curve_y']  # Standardabweichung als Mittelwert der Einzelkurven

    plt.figure(figsize=(12, 7))
    x_axis = np.arange(len(avg_curve))

    # Durchschnittskurve
    plt.plot(x_axis, avg_curve, label='Durchschnittskurve', color='royalblue', linewidth=2)

    # Schattierte Fläche ±1σ
    plt.fill_between(x_axis, avg_curve - std_curve, avg_curve + std_curve, 
                     color='royalblue', alpha=0.2)

    # Dummy-Plot für Legende, um Standardabweichung als Text zu zeigen
    plt.plot([], [], color='royalblue', linestyle='-', alpha=0.2, linewidth=10, 
              label=f'±1σ Bereich')

    # Layout-Optimierung
    plt.xlabel('Zeit / Index', fontsize=13)
    plt.ylabel('Druck / Wert', fontsize=13)
    plt.title(f'Variabilität (±1σ) Kategorie {category_selection}', fontsize=15)
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    plt.legend(loc='best', fontsize=11, frameon=True, fancybox=True)
    plt.tight_layout()
    plt.show()




def plot_process_analysis(df, category_selection, features_df):
    category_selection = int(category_selection)
    df_cat = df[df['Kategorie'] == category_selection]
    curves = [df_cat[df_cat['cu_id'] == cu_id]['y_achse'].values 
               for cu_id in df_cat['cu_id'].unique()]
    min_len = min(map(len, curves))
    curves_aligned = [c[:min_len] for c in curves]
    avg_curve = np.mean(curves_aligned, axis=0)

    feature_means = features_df[features_df['Kategorie'] == category_selection].mean()
    signal_energy = feature_means['signal_energy']
    total_length = int(feature_means['total_curve_length'])

    plt.figure(figsize=(12, 7))
    x_axis = np.arange(len(avg_curve))

    # Durchschnittskurve
    plt.plot(x_axis, avg_curve, label='Durchschnittskurve', color='royalblue', linewidth=2)

    # Endwert markieren
    end_x = x_axis[-1]
    end_y = avg_curve[-1]
    plt.scatter(end_x, end_y, color='black', s=80, marker='x', zorder=5)
    plt.scatter(0, 0, color='black', s=80, marker='x', zorder=5)

    # Horizontale Linie für die Länge (unterhalb der Kurve)
    y_min = np.min(avg_curve)
    y_offset = 0  # Abstand nach unten
    plt.hlines(y=y_offset, xmin=0, xmax=total_length, color='dimgray', linewidth=2)
    plt.text(
        total_length / 2, y_offset - 0.05 * (np.max(avg_curve) - y_min), 
        f'Länge: {total_length}', 
        ha='center', fontsize=11, fontweight='bold', color='dimgray'
    )

    # Dummy-Plots für Legende
    plt.plot([], [], color='black', linestyle='None', label=f'Signalenergie (∑y²): {signal_energy:.2f}')
    plt.plot([], [], color='black', linestyle='solid', label=f'Länge: {total_length}')

    # Plot-Layout
    plt.xlabel('Zeit / Index', fontsize=13)
    plt.ylabel('Druck / Wert', fontsize=13)
    plt.title(f'Prozessanalyse Kategorie {category_selection}', fontsize=15)
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    plt.legend(loc='best', fontsize=11, frameon=True, fancybox=True)
    plt.tight_layout()
    plt.show()



def manual_plot_selection(df, category_selection, features_df):
    plot_functions = {
        '1': ('Formanalyse (Max, Min, Diff)', plot_form_analysis),
        '2': ('Trend- & Symmetrieanalyse', plot_trend_symmetry),
        '3': ('Variabilität (±1σ)', plot_variability),
        '4': ('Prozessanalyse (Signalenergie, Kurvenlänge)', plot_process_analysis),
        '0': ('Beenden', None)
    }

    while True:
        print("\n📊 Verfügbare Plot-Optionen:")
        for key, (desc, _) in plot_functions.items():
            print(f"  {key}: {desc}")

        choice = input("Bitte wähle den Plot-Typ (Zahl eingeben): ")
        if choice not in plot_functions:
            print("⚠️  Ungültige Eingabe. Bitte erneut versuchen.")
            continue
        if choice == '0':
            print("🚪 Plot-Auswahl beendet.")
            break

        _, plot_func = plot_functions[choice]
        plot_func(df, category_selection, features_df)



def main():
    excel_file = "/Users/huppimarc/Library/CloudStorage/OneDrive-ZHAW/6ixers/PM4/DOE_Gesamtauswertung_verarbeitet.xlsx"

    xls = pd.ExcelFile(excel_file)
    doe_sheets = [sheet for sheet in xls.sheet_names if "DOE" in sheet]

    print("📂 Verfügbare DOE-Datensätze (Sheets):")
    for idx, sheet in enumerate(doe_sheets):
        print(f"  {idx+1}: {sheet}")

    sheet_choice = input(f"Bitte wähle den Datensatz (1-{len(doe_sheets)}): ")
    while not sheet_choice.isdigit() or int(sheet_choice) not in range(1, len(doe_sheets) + 1):
        sheet_choice = input(f"Ungültige Eingabe. Bitte wähle den Datensatz (1-{len(doe_sheets)}): ")

    selected_sheet = doe_sheets[int(sheet_choice) - 1]

    print("\n📈 Verfügbare Kategorien: 1, 2, 3, 4, 5, 6 oder 'all'")
    category_selection = input("Bitte wähle die Kategorie (1-6 oder 'all'): ")

    while category_selection not in ['1', '2', '3', '4', '5', '6', 'all']:
        category_selection = input("Ungültige Eingabe. Bitte wähle die Kategorie (1-6 oder 'all'): ")

    df = load_doe_data(excel_file, selected_sheet)

    features_df = extract_features(df, category_selection)
    print("\n✅ --- Berechnete Features ---")
    print(features_df)


    if category_selection != 'all':
        manual_plot_selection(df, category_selection, features_df)


if __name__ == "__main__":
    main()
