import textgrid
import json

# Lista de vocales con y sin acento
vocales = ["a", "e", "i", "o", "u", "á", "é", "í", "ó", "ú"]

def extract_vowel_times_from_letters(textgrid_path):
    # Cargar el archivo TextGrid
    tg = textgrid.TextGrid.fromFile(textgrid_path)
    
    vowel_data = []

    # Iterar sobre los tiers en el TextGrid
    for tier in tg:
        # Iterar sobre los intervalos de fonemas
        for interval in tier.intervals:
            # Filtrar las vocales (con y sin acento) de cada fonema
            if any(vowel in interval.mark.lower() for vowel in vocales):
                word = interval.mark.lower()
                word_start = round(interval.minTime, 2)
                word_end = round(interval.maxTime, 2)

                # Si el fonema es una vocal, distribuimos el tiempo de ese fonema entre las letras
                for i, char in enumerate(word):
                    if char in vocales:
                        # Calculamos el tiempo de inicio y fin de cada vocal
                        # Aproximación: distribuye el tiempo entre las letras de la palabra
                        char_start = word_start + (i / len(word)) * (word_end - word_start)
                        char_end = word_start + ((i + 3) / len(word)) * (word_end - word_start)

                        vowel_data.append({
                            "vocal": char,
                            "start": round(char_start, 2),
                            "end": round(char_end, 2)
                        })

    return vowel_data

# Ruta al archivo TextGrid generado por MFA
textgrid_path = "mfa/output/audio.TextGrid"

# Extraer las vocales y sus tiempos a nivel de letra
vowel_data = extract_vowel_times_from_letters(textgrid_path)

# Convertir a formato JSON
vowel_json = json.dumps(vowel_data, indent=4)

# Guardar el JSON con los tiempos de las vocales
with open('vocales_tiempos.json', 'w') as file:
    file.write(vowel_json)

# Imprimir el JSON generado
print(vowel_json)
