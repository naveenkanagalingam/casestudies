import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class PCAVisualizer:
    def __init__(self, filepath, category_col='kategorie', output_dir='PCA'):  # <–– Ausgabeordner ist jetzt "PCA"
        self.filepath = filepath
        self.category_col = category_col
        self.output_dir = output_dir
        self.filename = os.path.splitext(os.path.basename(filepath))[0]
        self.df = None
        self.X = None
        self.y = None
        self.X_scaled = None
        self.X_pca = None

        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        self.df = pd.read_excel(self.filepath)

        # Kategorie in binär umwandeln: 1 = akzeptabel (gut), alles andere = nicht akzeptabel (schlecht)
        self.df['Klasse'] = self.df[self.category_col].apply(lambda x: 1 if x == 1 else 2)

        self.y = self.df['Klasse']
        self.X = self.df.drop(columns=[self.category_col, 'Klasse'])

    def preprocess(self):
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)

    def apply_pca(self, n_components=2):
        pca = PCA(n_components=n_components)
        self.X_pca = pca.fit_transform(self.X_scaled)
        return pca.explained_variance_ratio_

    def plot(self):
        df_plot = pd.DataFrame(self.X_pca, columns=['PC1', 'PC2'])
        df_plot['Klasse'] = self.y

        plt.figure(figsize=(8, 6))
        for label, color, name in zip([1, 2], ['green', 'red'], ['akzeptabel', 'nicht akzeptabel']):
            subset = df_plot[df_plot['Klasse'] == label]
            plt.scatter(subset['PC1'], subset['PC2'], label=name, alpha=0.7, c=color)

        plt.title(f'PCA Projektion – {self.filename}')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.legend()
        plt.grid(True)

        output_path = os.path.join(self.output_dir, f'PCA_{self.filename}.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Plot gespeichert: {output_path}")

    def run(self):
        self.load_data()
        self.preprocess()
        explained_variance = self.apply_pca()
        self.plot()
        return explained_variance

# ---------- Ausführung ----------
if __name__ == '__main__':
    excel_files = [
    'Korrelation/DOE1_features_reduced_features.xlsx',
    'Korrelation/DOE2_features_reduced_features.xlsx',
    'Korrelation/DOE3_features_reduced_features.xlsx'
]

    for file in excel_files:
        visualizer = PCAVisualizer(file)
        variance = visualizer.run()
        print(f"{file} – erklärte Varianz der ersten zwei Komponenten: {variance}")
