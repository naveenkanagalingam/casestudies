### --- Datenaufbereitung --- ###
from Datenaufbereitung import DOEConverter

input_path = "xlsx_files/DOEs.xlsx"
output_path = "xlsx_files/DOEs_aufbereitet.xlsx"

converter = DOEConverter(input_path=input_path, output_path=output_path)
converter.process_all_sheets()

### --- Explorative Datenanalyse --- ###
from Explorative_Datenanalyse import CurveAnalyzer

# Benutzerinteraktion
pfad = "xlsx_files/DOEs_aufbereitet.xlsx"
eingabe = input("Gib die Kategorien-Auswahl ein (z. B. 1,2 oder 1 2): ").replace(",", " ")
kategorien_auswahl = [int(k) for k in eingabe.split() if k.strip().isdigit()]

analyzer = CurveAnalyzer(pfad, kategorien_auswahl)
analyzer.ausführen()

### --- Feature Engineering --- ###
from Feature_Engineering import process_excel_with_multiple_sheets

input_excel = "xlsx_files/DOEs_aufbereitet_alle_sheets_neu.xlsx"
process_excel_with_multiple_sheets(input_excel)

### --- Korrelationsmatrix --- ###
from Korrelationsmatrizen import KorrelationsmatrixGenerator

excel_files = [
    'feature_excels/DOE1_features.xlsx',
    'feature_excels/DOE2_features.xlsx',
    'feature_excels/DOE3_features.xlsx'
]

# Manuell bevorzugte Features:
priorisierte_features = {
    "area_total", "global_maxima", "Trajectory_length", "slope_max", 'peak_count', 'valley_count'
}

if __name__ == "__main__":
    generator = KorrelationsmatrixGenerator(
        dateipfade=excel_files,
        korrelationsschwelle=0.9,
        priorisierte_features=priorisierte_features
    )
    generator.erstelle_korrelationen()

### --- ML Modell --- ###
from Machine_Learning_Models.Random_Forest_Classifier import MLModel
from Machine_Learning_Models.Random_Forest_Classifier import run_multiple_models


excel_files = [
    'Korrelation/DOE1_features_reduced_features.xlsx',
    'Korrelation/DOE2_features_reduced_features.xlsx',
    'Korrelation/DOE3_features_reduced_features.xlsx'
]

run_multiple_models(excel_files)

### --- Logistische Regression --- ###
from Machine_Learning_Models.Logistische_Regression import LogisticRegressionModel
from Machine_Learning_Models.Logistische_Regression import run_multiple_logistic_models

excel_files = [
    'Korrelation/DOE1_features_reduced_features.xlsx',
    'Korrelation/DOE2_features_reduced_features.xlsx',
    'Korrelation/DOE3_features_reduced_features.xlsx'
]

run_multiple_logistic_models(excel_files)

### --- Gradient Boosting Model --- ###
from Machine_Learning_Models.GradientBoostingModel import GradientBoostingModel
from Machine_Learning_Models.GradientBoostingModel import run_multiple_gradient_boosting_models

excel_files = [
    'Korrelation/DOE1_features_reduced_features.xlsx',
    'Korrelation/DOE2_features_reduced_features.xlsx',
    'Korrelation/DOE3_features_reduced_features.xlsx'
]

run_multiple_gradient_boosting_models(excel_files)
