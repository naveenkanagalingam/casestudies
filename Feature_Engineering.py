import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class KorrelationsmatrixGenerator:
    def __init__(self, dateipfade, output_dir="Korrelation", korrelationsschwelle=0.9, priorisierte_features=None):
        self.dateipfade = dateipfade
        self.output_dir = output_dir
        self.korrelationsschwelle = korrelationsschwelle
        self.priorisierte_features = set(priorisierte_features) if priorisierte_features else set()
        os.makedirs(self.output_dir, exist_ok=True)

    def erstelle_korrelationen(self):
        for datei in self.dateipfade:
            self._verarbeite_datei(datei)

    def _verarbeite_datei(self, pfad):
        if not os.path.exists(pfad):
            print(f"Datei nicht gefunden: {pfad}")
            return

        try:
            df = pd.read_excel(pfad)
        except Exception as e:
            print(f"Fehler beim Lesen von {pfad}: {e}")
            return

        sheetname = os.path.splitext(os.path.basename(pfad))[0]

        exclude_cols = {"cu_id", "kategorie"}
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        numeric_df = df[feature_cols].select_dtypes(include='number')

        # 1. Korrelationsmatrix berechnen
        df_corr = numeric_df.corr().abs()

        # 2. Speichere ursprüngliche Korrelationsmatrix als Bild
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title(f"Korrelationsmatrix – {sheetname}")
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f"{sheetname}_correlation_matrix.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Korrelationsmatrix gespeichert: {plot_path}")

        # 3. Hoch korrelierte Paare erkennen und Auswahl treffen
        to_drop = set()
        entscheidungen = []  # Liste für Logging
        columns = list(df_corr.columns)

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1 = columns[i]
                col2 = columns[j]

                if col1 in to_drop or col2 in to_drop:
                    continue

                corr_val = df_corr.loc[col1, col2]
                if corr_val > self.korrelationsschwelle:
                    if col1 in self.priorisierte_features and col2 in self.priorisierte_features:
                        entscheidungen.append({
                            "feature_1": col1,
                            "feature_2": col2,
                            "korrelation": corr_val,
                            "aktion": "beide priorisiert – behalten"
                        })
                        continue
                    elif col1 in self.priorisierte_features:
                        to_drop.add(col2)
                        entscheidungen.append({
                            "feature_1": col1,
                            "feature_2": col2,
                            "korrelation": corr_val,
                            "aktion": f"{col2} gelöscht (nicht priorisiert)"
                        })
                    elif col2 in self.priorisierte_features:
                        to_drop.add(col1)
                        entscheidungen.append({
                            "feature_1": col1,
                            "feature_2": col2,
                            "korrelation": corr_val,
                            "aktion": f"{col1} gelöscht (nicht priorisiert)"
                        })
                    else:
                        std1 = numeric_df[col1].std()
                        std2 = numeric_df[col2].std()
                        less_important = col1 if std1 < std2 else col2
                        if less_important not in self.priorisierte_features:
                            to_drop.add(less_important)
                            entscheidungen.append({
                                "feature_1": col1,
                                "feature_2": col2,
                                "korrelation": corr_val,
                                "aktion": f"{less_important} gelöscht (geringere Standardabweichung)"
                            })
                        else:
                            entscheidungen.append({
                                "feature_1": col1,
                                "feature_2": col2,
                                "korrelation": corr_val,
                                "aktion": "nicht gelöscht (weniger wichtiges Feature ist priorisiert)"
                            })

        # 4. Reduziertes DataFrame erstellen
        reduced_df = numeric_df.drop(columns=to_drop)

        # 5. Reduziertes Feature-Set als Excel speichern
        reduced_path = os.path.join(self.output_dir, f"{sheetname}_reduced_features.xlsx")
        reduced_df.to_excel(reduced_path, index=False)
        print(f"Reduzierte Features gespeichert: {reduced_path}")
        if to_drop:
            print(f"Entfernte Features ({len(to_drop)}): {sorted(to_drop)}")
        else:
            print("Keine korrelierten Features über der Schwelle gefunden.")

        # 6. Entscheidungs-Log speichern
        if entscheidungen:
            entscheidungs_df = pd.DataFrame(entscheidungen)
            log_path = os.path.join(self.output_dir, f"{sheetname}_korrelation_entfernt.csv")
            entscheidungs_df.to_csv(log_path, index=False)
            print(f"Entscheidungsübersicht gespeichert: {log_path}")

# Beispiel-Dateien
excel_files = [
    'feature_excels/DOE1_features.xlsx',
    'feature_excels/DOE2_features.xlsx',
    'feature_excels/DOE3_features.xlsx'
]

# Manuell bevorzugte Features
priorisierte_features = {
    "area_total", "global_maxima", "Trajectory_length", "slope_max", "peak_count", "valley_count"
}

if __name__ == "__main__":
    generator = KorrelationsmatrixGenerator(
        dateipfade=excel_files,
        korrelationsschwelle=0.7,
        priorisierte_features=priorisierte_features
    )
    generator.erstelle_korrelationen()
