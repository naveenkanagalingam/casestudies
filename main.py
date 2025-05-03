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
from Feature_Engineering import FeatureEngineer

# 1. Initialisieren
engineer = FeatureEngineer("xlsx_files/DOEs_aufbereitet_alle_sheets_neu.xlsx")

# 2. Schritte durchführen
engineer.extract_features()
engineer.apply_clustering()
engineer.reduce_correlated_features()
engineer.save_features("xlsx_files/alle_formparameter_finale.csv")

### --- ML Modell --- ###
from MML import MLModel

# Pfad zur CSV anpassen
csv_file = "xlsx_files/alle_formparameter_finale.csv"

# Modellobjekt erzeugen und Pipeline ausführen
model = MLModel(csv_path=csv_file)
model.run_all()
