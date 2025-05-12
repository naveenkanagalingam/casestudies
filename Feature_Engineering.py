import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import os

def calculate_features(df_group):
    y = df_group['y_achse'].values
    x = df_group['x_achse'].values

    area_total = float(np.trapz(y, x))
    signal_energy = float(np.sum(np.square(y))) 
    global_maxima = float(np.max(y))
    global_minima = float(np.min(y))
    mean_y = float(np.mean(y))
    median_y = float(np.median(y))
    sd = float(np.std(y))
    trajectory_length = float(np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)))
    peak_count = int(len(find_peaks(y)[0]))
    valley_count = int(len(find_peaks(-y)[0]))

    slope = np.gradient(y, x)
    slope_max = float(np.max(slope))
    slope_min = float(np.min(slope))

    q = len(y) // 4
    slope_phase1 = float(np.mean(slope[:q]))
    slope_phase2 = float(np.mean(slope[q:2*q]))
    slope_phase3 = float(np.mean(slope[2*q:3*q]))
    slope_phase4 = float(np.mean(slope[3*q:]))

    features = {
        'area_total': area_total,
        'signal_energy': signal_energy,             
        'global_maxima': global_maxima,
        'global_minima': global_minima,
        'mean_y': mean_y,
        'median_y': median_y,
        'sd': sd,
        'trajectory_length': trajectory_length,
        'maxima_count': peak_count,
        'minima_count': valley_count,
        'slope_max': slope_max,
        'slope_min': slope_min,
        'slope_phase1': slope_phase1,
        'slope_phase2': slope_phase2,
        'slope_phase3': slope_phase3,
        'slope_phase4': slope_phase4,
        'mpv_injection_speed': float(df_group['mpv_injection_speed'].mean()),
        'mpv_holding_pressure': float(df_group['mpv_holding_pressure'].mean()),
        'mpv_mold_temp': float(df_group['mpv_mold_temp'].mean()),
        'cu_id': df_group['cu_id'].iloc[0],
        'kategorie': df_group['Kategorie'].iloc[0]
    }

    return features

def process_excel_with_multiple_sheets(input_excel_path, output_folder='feature_excels'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Zielordner erstellt: {output_folder}")
    else:
        print(f"Zielordner vorhanden: {output_folder}")

    all_sheets = pd.read_excel(input_excel_path, sheet_name=None)

    for sheet_name, df in all_sheets.items():
        print(f"Verarbeite Sheet: {sheet_name}")

        required_cols = {'cu_id', 'x_achse', 'y_achse'}
        if not required_cols.issubset(df.columns):
            print(f"Übersprungen: '{sheet_name}' enthält nicht alle benötigten Spalten {required_cols}")
            continue

        features_list = []
        for cu_id, group_df in df.groupby('cu_id'):
            feature_row = calculate_features(group_df)
            features_list.append(feature_row)

        if features_list:
            feature_df = pd.DataFrame(features_list)
            feature_df = feature_df.apply(pd.to_numeric, errors='ignore')

            output_path = os.path.join(output_folder, f"{sheet_name}_features.xlsx")
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                feature_df.to_excel(writer, index=False, sheet_name="Features")

            print(f"Excel-Datei gespeichert: {output_path}")
        else:
            print(f"Keine gültigen Daten in Sheet '{sheet_name}'.")

# Nur zur direkten Ausführung
if __name__ == "__main__":
    input_excel = "xlsx_files/DOEs_aufbereitet_alle_sheets_neu.xlsx"
    process_excel_with_multiple_sheets(input_excel)
