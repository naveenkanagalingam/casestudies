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


class FeatureEngineer:
    def __init__(self, excel_path, protected_features=None, correlation_threshold=0.7):
        self.excel_path = excel_path
        self.df_raw = pd.read_excel(excel_path)
        self.df_features = None
        self.protected_features = protected_features or {"global_max_x", "global_min_x"}
        self.correlation_threshold = correlation_threshold

    def extract_features(self):
        kurven_dict = {
            cu_id: gruppe.sort_values(by="x_achse")[["x_achse", "y_achse"]].rename(columns={"x_achse": "x", "y_achse": "y"})
            for cu_id, gruppe in self.df_raw.groupby("cu_id")
        }

        feature_list = []

        for cu_id, df_curve in kurven_dict.items():
            x = df_curve["x"].values
            y = df_curve["y"].values

            if len(x) < 5:
                continue

            kategorie = self.df_raw[self.df_raw["cu_id"] == cu_id]["Kategorie"].iloc[0]

            # Feature-Berechnung
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

        self.df_features = pd.DataFrame(feature_list)

    def apply_clustering(self, n_clusters=3):
        feature_cols = [col for col in self.df_features.columns if col not in {"cu_id", "kategorie"}]
        features_scaled = StandardScaler().fit_transform(self.df_features[feature_cols])

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df_features["cluster"] = kmeans.fit_predict(features_scaled)

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features_scaled)
        self.df_features["PCA1"] = pca_result[:, 0]
        self.df_features["PCA2"] = pca_result[:, 1]

        # Visualisierung
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=self.df_features, x="PCA1", y="PCA2", hue="cluster", palette="Set1")
        plt.title("KMeans-Clustering der Kurven (PCA 2D)")
        plt.grid(True)
        plt.legend(title="Cluster")
        plt.tight_layout()
        plt.show()

    def reduce_correlated_features(self):
        exclude_cols = {"cu_id", "cluster", "PCA1", "PCA2", "kategorie"}
        feature_cols = [col for col in self.df_features.columns if col not in exclude_cols]
        corr = self.df_features[feature_cols].corr().abs()

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

            group_variances = self.df_features[keep_candidates].var()
            best_col = group_variances.idxmax()

            for col in group:
                if col != best_col and col not in self.protected_features:
                    to_drop.add(col)

        self.df_features = self.df_features.drop(columns=to_drop)
        print("✅ Entfernte korrelierte Features:", sorted(to_drop))

    def save_features(self, path="xlsx_files/alle_formparameter_final.csv"):
        self.df_features.to_csv(path, sep=";", index=False)
        print(f"✅ Feature-Datei gespeichert unter: {path}")



