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
from Feature_Engineering import FeatureEngineerMultiSheet

pfad = "xlsx_files/DOEs_aufbereitet.xlsx"
engineer = FeatureEngineerMultiSheet(excel_path=pfad)
engineer.process_all_sheets()

### --- ML Modell --- ###
from MML import MLModel
from MML import run_multiple_models


csv_files = [
    "xlsx_files/features/DOE1_features.csv",
    "xlsx_files/features/DOE2_features.csv",
    "xlsx_files/features/DOE3_features.csv"
]

run_multiple_models(csv_files)

