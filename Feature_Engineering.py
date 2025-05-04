import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.metrics import auc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os

class FeatureEngineerMultiSheet:
    def __init__(self, excel_path, protected_features=None, correlation_threshold=0.7,
                 output_dir="xlsx_files/features", plot_dir="Cluster_PCA_Plots"):
        self.excel_path = excel_path
        self.sheets_raw = pd.read_excel(excel_path, sheet_name=None)
        self.protected_features = protected_features or {"global_max_x", "global_min_x"}
        self.correlation_threshold = correlation_threshold
        self.output_dir = output_dir
        self.plot_dir = plot_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

    def process_all_sheets(self):
        all_feature_data = []

        for sheetname, df_raw in self.sheets_raw.items():
            print(f"\nBearbeite Sheet: {sheetname}")
            df_features = self.extract_features(df_raw)
            if df_features.empty:
                print(f"Überspringe Sheet {sheetname} – zu wenige gültige Kurven.")
                continue

            df_features = self.apply_clustering(df_features, sheetname)
            df_features = self.reduce_correlated_features(df_features, sheetname)

            # NEU: Speichere sheetweise Korrelationsmatrix
            self.save_correlation_matrix_per_sheet(df_features, sheetname)

            # Speichern
            output_path = os.path.join(self.output_dir, f"{sheetname}_features.csv")
            df_features.to_csv(output_path, sep=";", index=False)
            print(f"Gespeichert: {output_path}")

            all_feature_data.append(df_features.copy())

        # Optional: Globale Korrelationsmatrix (alle Sheets kombiniert)
        if all_feature_data:
            df_all_combined = pd.concat(all_feature_data, ignore_index=True)
            self.save_correlation_matrix_from_combined(df_all_combined)

    def extract_features(self, df_raw):
        feature_list = []

        for cu_id, gruppe in df_raw.groupby("cu_id"):
            df_curve = gruppe.sort_values(by="x_achse")[["x_achse", "y_achse"]].rename(columns={"x_achse": "x", "y_achse": "y"})
            x = df_curve["x"].values
            y = df_curve["y"].values

            if len(x) < 5:
                continue

            kategorie = gruppe["Kategorie"].iloc[0]

            x_center = (x.min() + x.max()) / 2
            max_x = x[np.argmax(y)]
            symmetry_deviation = (max_x - x_center) / (x.max() - x.min())
            curve_skewness = skew(y)
            area_total = auc(x, y)

            peak_idx = np.argmax(y)
            if 1 < peak_idx < len(y) - 2:
                slope_rising = (y[peak_idx] - y[peak_idx - 2]) / (x[peak_idx] - x[peak_idx - 2])
                slope_falling = (y[peak_idx + 2] - y[peak_idx]) / (x[peak_idx + 2] - x[peak_idx])
            else:
                slope_rising = np.nan
                slope_falling = np.nan

            global_max_x = x[np.argmax(y)]
            global_min_x = x[np.argmin(y)]
            peak_value = np.max(y)
            peak_x = x[np.argmax(y)]
            min_value = np.min(y)
            min_x = x[np.argmin(y)]
            peak_to_min_diff = peak_value - min_value

            feature_list.append({
                "cu_id": cu_id,
                "kategorie": kategorie,
                "symmetry_deviation": symmetry_deviation,
                "curve_skewness": curve_skewness,
                "area_total": area_total,
                "slope_rising": slope_rising,
                "slope_falling": slope_falling,
                "global_max_x": global_max_x,
                "global_min_x": global_min_x,
                "Top1_Peak_Druck": peak_value,
                "Top1_Peak_Position_x": peak_x,
                "Top1_Minima_Druck": min_value,
                "Top1_Minima_Position_x": min_x,
                "Peak_to_Minima_Diff": peak_to_min_diff
            })

        return pd.DataFrame(feature_list)

    def apply_clustering(self, df_features, sheetname, n_clusters=3):
        feature_cols = [col for col in df_features.columns if col not in {"cu_id", "kategorie"}]
        features_scaled = StandardScaler().fit_transform(df_features[feature_cols])

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df_features["cluster"] = kmeans.fit_predict(features_scaled)

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features_scaled)
        df_features["PCA1"] = pca_result[:, 0]
        df_features["PCA2"] = pca_result[:, 1]

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_features, x="PCA1", y="PCA2", hue="cluster", palette="Set1")
        plt.title(f"KMeans-Clustering – {sheetname}")
        plt.grid(True)
        plt.tight_layout()

        plot_path = os.path.join(self.plot_dir, f"{sheetname}_PCA_Cluster.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"PCA-Plot gespeichert: {plot_path}")

        return df_features

    def reduce_correlated_features(self, df_features, sheetname):
        exclude_cols = {"cu_id", "cluster", "PCA1", "PCA2", "kategorie"}
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        corr = df_features[feature_cols].corr().abs()

        correlated_groups = defaultdict(set)
        for i in range(len(corr.columns)):
            for j in range(i):
                if corr.iloc[i, j] > self.correlation_threshold:
                    col1 = corr.columns[i]
                    col2 = corr.columns[j]
                    correlated_groups[col1].add(col2)
                    correlated_groups[col2].add(col1)

        processed = set()
        to_drop = set()

        for base_col in correlated_groups:
            if base_col in processed:
                continue

            group = {base_col, *correlated_groups[base_col]}
            for col in list(group):
                group.update(correlated_groups[col])

            group = list(group)
            processed.update(group)

            keep_candidates = [col for col in group if col not in self.protected_features]
            if not keep_candidates:
                continue

            group_variances = df_features[keep_candidates].var()
            best_col = group_variances.idxmax()

            for col in group:
                if col != best_col and col not in self.protected_features:
                    to_drop.add(col)

        df_reduced = df_features.drop(columns=to_drop)
        print(f"Entfernte korrelierte Features im Sheet {sheetname}:", sorted(to_drop))
        return df_reduced

    def save_correlation_matrix_per_sheet(self, df_features, sheetname):
        exclude_cols = {"cu_id", "cluster", "PCA1", "PCA2", "kategorie"}
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        corr = df_features[feature_cols].corr().abs()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title(f"Korrelationsmatrix – {sheetname}")
        plt.tight_layout()

        plot_path = os.path.join(self.plot_dir, f"{sheetname}_correlation_matrix.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Korrelationsmatrix gespeichert: {plot_path}")

    def save_correlation_matrix_from_combined(self, df_combined):
        exclude_cols = {"cu_id", "cluster", "PCA1", "PCA2", "kategorie"}
        feature_cols = [col for col in df_combined.columns if col not in exclude_cols]
        corr = df_combined[feature_cols].corr().abs()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title("Globale Korrelationsmatrix – alle Sheets kombiniert")
        plt.tight_layout()

        plot_path = os.path.join(self.plot_dir, "GLOBAL_correlation_matrix.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Globale Korrelationsmatrix gespeichert: {plot_path}")


if __name__ == "__main__":
    pfad = "xlsx_files/DOEs_aufbereitet_alle_sheets_neu.xlsx"
    engineer = FeatureEngineerMultiSheet(excel_path=pfad)
    engineer.process_all_sheets()
