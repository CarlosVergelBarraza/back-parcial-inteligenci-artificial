import csv
import json

def csv_to_json(csv_file, json_file):
    # Abrir el archivo CSV
    with open(csv_file, 'r') as csv_input:
        # Leer el archivo CSV
        csv_reader = csv.DictReader(csv_input)
        
        # Convertir el CSV a una lista de diccionarios
        data = list(csv_reader)
        
        # Escribir el JSON
        with open(json_file, 'w') as json_output:
            json.dump(data, json_output, indent=2)

# Reemplaza 'input.csv' con el nombre de tu archivo CSV y 'output.json' con el nombre que desees para el archivo JSON
csv_to_json('resultado_de_merge.csv', 'output.json')
