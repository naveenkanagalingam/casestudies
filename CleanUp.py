import os
import shutil

class FolderCleaner:
    def __init__(self, base_path="."):
        self.base_path = base_path
        self.target_folders = [
            "Klassifikationsberichte",
            "Cluster_PCA_Plots",
            "Konfusionsmatrix",
            "plots",
            "xlsx_files",
            "Plots_DTW_Verteilungen",
            "Plots_Heatmaps",
            "Plots_Improved",
            'extended_features',
            'features',
            'Korrelation',
            'feature_excels'
        ]
        self.exclude_files = {"DOEs.xlsx"}

    def clear_all(self):
        for folder in self.target_folders:
            self.clear_folder(folder)

    def clear_folder(self, folder_name):
        folder_path = os.path.join(self.base_path, folder_name)

        if not os.path.exists(folder_path):
            print(f"[Ignoriert] Ordner nicht gefunden: {folder_path}")
            return

        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)

            if item in self.exclude_files:
                print(f"[Ausgenommen] {item_path}")
                continue

            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)
                    print(f"[Datei gelöscht] {item_path}")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"[Ordnerinhalt gelöscht] {item_path}")
            except Exception as e:
                print(f"[Fehler] {item_path}: {e}")

if __name__ == "__main__":
    cleaner = FolderCleaner()
    cleaner.clear_all()
