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
        for i in range(len(df_corr.columns)):
            for j in range(i + 1, len(df_corr.columns)):
                col1 = df_corr.columns[i]
                col2 = df_corr.columns[j]

                # Nur vergleichen, wenn beide Spalten noch aktiv sind
                if col1 in to_drop or col2 in to_drop:
                    continue

                if df_corr.iloc[i, j] > self.korrelationsschwelle:
                    if col1 in self.priorisierte_features and col2 not in self.priorisierte_features:
                        to_drop.add(col2)
                    elif col2 in self.priorisierte_features and col1 not in self.priorisierte_features:
                        to_drop.add(col1)
                    else:
                        std1 = numeric_df[col1].std()
                        std2 = numeric_df[col2].std()
                        less_important = col1 if std1 < std2 else col2
                        to_drop.add(less_important)

        # 4. Reduziertes DataFrame erstellen und wichtige Spalten wieder anhängen
        reduced_df = numeric_df.drop(columns=to_drop)

        # cu_id und kategorie wieder einfügen, falls sie im Original vorhanden sind
        meta_cols = []
        for col in ["cu_id", "kategorie"]:
            if col in df.columns:
                meta_cols.append(df[col])

        if meta_cols:
            reduced_df = pd.concat([*meta_cols, reduced_df], axis=1)

        # 5. Reduziertes Feature-Set als Excel speichern
        reduced_path = os.path.join(self.output_dir, f"{sheetname}_reduced_features.xlsx")
        reduced_df.to_excel(reduced_path, index=False)


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
        korrelationsschwelle=0.9,
        priorisierte_features=priorisierte_features
    )
    generator.erstelle_korrelationen()
