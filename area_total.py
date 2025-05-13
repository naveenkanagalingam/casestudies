import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class CurveAnalyzer:
    def __init__(self, excel_path, output_dir="Integral"):
        self.excel_path = excel_path
        self.output_dir = output_dir
        self.sheets = pd.read_excel(excel_path, sheet_name=None)

        # Ordner erstellen, falls nicht vorhanden
        os.makedirs(self.output_dir, exist_ok=True)

    def process_all_sheets(self):
        for sheet_name, df in self.sheets.items():
            print(f"🔍 Bearbeite Sheet: {sheet_name}")
            self.process_single_sheet(df, sheet_name)

    def process_single_sheet(self, df, sheet_name):
        category_1_curves = self.get_curves_by_category(df, category=1)
        avg_curve_1, x_axis = self.compute_average_curve(category_1_curves)

        # Durchschnittskurve von Kategorie 3 als Hintergrund
        background_curves = self.get_curves_by_category(df, category=3)
        aligned_background = self.align_curves(background_curves, len(avg_curve_1))
        mean_bg_curve = np.mean(aligned_background, axis=0) if aligned_background else None

        self.plot_average_with_background(
            avg_curve_1, x_axis, mean_bg_curve, sheet_name
        )

    def get_curves_by_category(self, df, category):
        return [
            group.sort_values("x_achse")[["x_achse", "y_achse"]].values
            for cu_id, group in df[df["Kategorie"] == category].groupby("cu_id")
        ]

    def compute_average_curve(self, curves):
        min_len = min(len(c) for c in curves)
        aligned = [curve[:min_len] for curve in curves]
        x_axis = aligned[0][:, 0]
        y_matrix = np.array([c[:, 1] for c in aligned])
        avg_y = np.mean(y_matrix, axis=0)
        return avg_y, x_axis

    def align_curves(self, curves, target_len):
        return [c[:target_len, 1] for c in curves if len(c) >= target_len]

    def plot_average_with_background(self, avg_curve, x_axis, bg_curve, sheet_name):
        area = np.trapz(avg_curve, x_axis)

        plt.figure(figsize=(10, 5))

        # Hintergrund: Durchschnitt von Kategorie 3 (falls vorhanden)
        if bg_curve is not None:
            plt.plot(x_axis, bg_curve, color='gray', linewidth=2, alpha=0.8, label='Durchschnitt Kat. 3')

        # Durchschnittskurve Kat. 1
        plt.plot(x_axis, avg_curve, label='Durchschnitt Kat. 1', color='blue', linewidth=2)
        plt.fill_between(x_axis, avg_curve, color='blue', alpha=0.3, label=f'Fläche = {area:.2f}')

        plt.xlabel('x-Achse')
        plt.ylabel('y-Achse')
        plt.title(f'Integral der Durchschnittskurve von Kategorie 1\n(Hintergrund: Kat. 3)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, f"{sheet_name}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"Plot gespeichert unter: {plot_path}")

if __name__ == "__main__":
    excel_path = "xlsx_files/DOEs_aufbereitet.xlsx"
    analyzer = CurveAnalyzer(excel_path)
    analyzer.process_all_sheets()
