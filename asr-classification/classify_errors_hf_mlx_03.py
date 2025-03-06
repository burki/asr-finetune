import pandas as pd
import re
import ast
import time
from pathlib import Path
from langchain.prompts import PromptTemplate
from mlx_lm import load, generate

def normalize(text):
    """
    Entfernt bestimmte Zeichen aus dem Text und konvertiert ihn in Kleinbuchstaben.
    """
    def process_single_text(single_text):
        result = single_text.strip().lower()
        result = re.sub(r"[!?.,\\]\\[\\\\;']", "", result)
        return result

    if isinstance(text, list):
        return [process_single_text(t) for t in text]
    elif isinstance(text, str):
        return process_single_text(text)
    else:
        raise TypeError("Input must be a string or a list of strings.")

# Create output directory if it doesn't exist
output_dir = Path("asr-classification/outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# Load dataset (CSV or JSON fallback)
finetuned = False
model_tag = 'small_eg_fzh_combined_v2_transformers' if finetuned else 'small_eg_fzh_combined_v2_whisper'

file_path_json = Path("asr-classification") / f"{model_tag}.json"
file_path_csv = Path("asr-classification") / f"{model_tag}_error_types.csv"

# Load main data
try:
    data = pd.read_json(file_path_json)
    print(f"===> JSON-Input geladen: {file_path_json}")
except Exception as e:
    print(f"Fehler beim Laden der JSON: {e}")
    try:
        data = pd.read_csv(file_path_csv)
        print(f"===> CSV-Input geladen: {file_path_csv}")
    except Exception as e:
        print(f"Fehler beim Laden der CSV: {e}")
        raise

# Save original column order
original_columns = list(data.columns)
print("===> Originale Spaltennamen:", original_columns)

# Initialize results DataFrame with same index as data
results = pd.DataFrame(index=data.index)

# Error Types (Fehlertypen) einlesen
try:
    df = pd.read_csv('asr-classification/error_types_new.csv')
    print("===> Fehlertypen geladen")
except Exception as e:
    print(f"Fehler beim Laden der Fehlertypen: {e}")
    raise

# Definiere das Sprachmodell
model_name = "mlx-community/Llama-3.3-70B-Instruct-4bit"
#model_name = ""

# Lade das Modell
#print("===> Lade Sprachmodell...")
model, tokenizer = load(model_name)
print(f"===> Sprachmodell geladen: {model_name}")

def classify_error_type(row, error_type, error_description, example, tag, idx, finetuned=False):
    """
    Klassifiziert Fehler im Transkriptions-Output mithilfe eines Sprachmodells.
    """
    try:
        original = normalize(row['original']) if finetuned else normalize(row['target']) # Original human transcript 
        prediction = normalize(row['predictions']) if finetuned else normalize(row['actual']) # ASR generated transcript 
    except (TypeError, KeyError) as e:
        print(f"===> Fehler bei der Textverarbeitung: {e}")
        return [0.0, 0, 'Fehler bei der Verarbeitung']

    # Prompt für LLM
    prompt = f"""Du erhältst zwei Transkripte derselben Sprachaufzeichnung zum Vergleich:
1. Ein Originaltranskript, das von einem Menschen erstellt wurde.
2. Ein ASR-Transkript, das durch ein Sprache-zu-Text-Modell durch automatische Spracherkennung (ASR) generiert wurde.

Deine Aufgabe ist es, das ASR-Transkript auf das Vorhandensein eines spezifischen Fehlertyps im Vergleich zum Originaltranskript zu prüfen.

Begründe in einem Satz, weshalb du einen Fehlertyp im ASR-Transkript erkannt oder nicht erkannt hast.
Nenne alle falschen beziehungsweise fehlenden Wörter des ASR-Transkripts. 
Kennzeichne jeden Fehler mit einem Apostroph ' direkt vor und nach dem Wort. 
Verwende ein einheitliches Formulierungsmuster für die Antwort. 

**WICHTIG**: Begrenze jede Antwort auf **maximal** 300 Zeichen, die **keinesfalls** überschritten werden dürfen. 

Bewerte den Fehlertyp wie folgt:
- Falls der Fehlertyp _mit absoluter Sicherheit_ auftritt, gib die Wahrscheinlichkeit 1 an.
- Falls der Fehlertyp _mit absoluter Sicherheit_ nicht auftritt, gib die Wahrscheinlichkeit 0 an.
- Falls keine absolute Sicherheit vorliegt, wähle einen Wert zwischen 0 und 1 mit einer Dezimalstelle, zum Beispiel 0.1 oder 0.2 und so weiter. 
- Wähle den Wert 0.5 bei kompletter Unsicherheit.

Zähle, wie oft der Fehlertyp im ASR-Transkript vorkommt (als ganze Zahl).

**BEACHTE**: Ein Eigenname ist eine sprachliche Bezeichnung, die zur eindeutigen Identifikation einer spezifischen Entität dient, einschließlich Personen, Orte, einzigartige Objekte, bedeutende Ereignisse oder klar definierte Themen.

Deine Antwort _muss_ **ausschließlich** als ein JSON-Objekt im folgenden Format erfolgen:
{{"p": "<Wahrscheinlichkeit>", "N": "<Anzahl>", "warum": "{error_type} {tag}: Begründung"}}

Antworte _ausschließlich_ mit diesem JSON-Objekt und gib keine zusätzlichen Kommentare, Erklärungen oder Formatierungen aus.

Originaltext: {original}
Transkription: {prediction}
Fehlertyp: {error_type}
Tag: {tag}
Beschreibung: {error_description}
Beispiel: {example}"""

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    try:
        response = generate(model, tokenizer, prompt=prompt, verbose=False)
        print(f"-> LLM Response für {tag} (Zeile {idx}): {response}")
        
        # Try parsing the response with better error handling
        try:
            # First attempt: direct parsing
            answer = ast.literal_eval(response)
        except (SyntaxError, ValueError) as json_err:
            print(f"JSON parsing failed, attempting fix: {json_err}")
            
            # Second attempt: Try to extract valid JSON with regex
            json_pattern = r'(\{\"p\"\s*:\s*\"[0-9.]+\"\s*,\s*\"N\"\s*:\s*\"[0-9]+\"\s*,\s*\"warum\"\s*:\s*\"[^\"]+\"\s*\})'
            match = re.search(json_pattern, response)
            if match:
                try:
                    answer = ast.literal_eval(match.group(1))
                except:
                    # If regex extraction fails, create a default response
                    answer = {"p": "0.0", "N": "0", "warum": f"Parsing error: {tag}"}
            else:
                # Third attempt: Try to fix common JSON errors
                fixed_response = response.replace('""', '"')  # Fix double quotes
                fixed_response = re.sub(r'([{\[,]\s*)"([^"]+)"\s*:', r'\1"\2":', fixed_response)  # Fix unquoted keys
                try:
                    # Try to find a JSON-like structure
                    json_start = fixed_response.find('{')
                    json_end = fixed_response.rfind('}')
                    if json_start >= 0 and json_end >= 0:
                        json_str = fixed_response[json_start:json_end+1]
                        answer = ast.literal_eval(json_str)
                    else:
                        # Create default response if no JSON-like pattern is found
                        answer = {"p": "0.0", "N": "0", "warum": f"Malformed response for {tag}"}
                except:
                    # Final fallback
                    answer = {"p": "0.0", "N": "0", "warum": f"Malformed response for {tag}"}
        
        # Convert types
        try:
            probability = float(answer["p"])
        except (ValueError, TypeError):
            probability = 0.0
            
        try:
            N_error = int(answer["N"])
        except (ValueError, TypeError):
            N_error = 0
            
        explain = answer.get("warum", f"Keine Erklärung für {tag}")
        
        return [probability, N_error, explain]
    except Exception as e:
        print(f"===> Fehler bei der LLM-Verarbeitung für {tag} (Zeile {idx}): {e}")
        return [0.0, 0, f'Fehler: {str(e)}']
    
# Liste für die neuen Spalten vorbereiten
new_columns = []

# Wende die Klassifizierung für die ersten X Zeilen von df (Error Types) an
for index, df_row in df.head(30).iterrows():
    start_time = time.time()

    error_type = df_row['Fehlertyp']
    description = df_row['Definition']
    example = df_row['Beispiel']
    tag = df_row['tag']

    # Spaltennamen für diesen Fehlertyp
    prob_col = f"{tag} p"
    count_col = f"{tag} N"
    warum_col = f"{tag} warum"
    
    # Füge neue Spalten zur Liste hinzu
    new_columns.extend([prob_col, count_col, warum_col])

    # Wähle hier aus, wie viele Zeilen/Segmente aus "data" (Transkript) klassifiziert werden sollen
    limit_rows = False # True: Nur x Zeilen werden ausgelesen, False: Alle Zeilen
    data_subset = data.head(2) if limit_rows else data

    print(f"\n===> Verarbeite Fehlertyp {tag} ({error_type})")

    # Process each row in the subset
    for idx, row in data_subset.iterrows():
        try:
            prob, count, reason = classify_error_type(
                row, 
                error_type=error_type, 
                error_description=description, 
                example=example, 
                tag=tag,
                idx=idx,
                finetuned=finetuned
            )
            
            # Store results
            results.at[idx, prob_col] = prob
            results.at[idx, count_col] = count
            results.at[idx, warum_col] = reason
            
        except Exception as e:
            print(f"===> Fehler bei der Verarbeitung von Zeile {idx}: {e}")
            results.at[idx, prob_col] = 0.0
            results.at[idx, count_col] = 0
            results.at[idx, warum_col] = f"Fehler: {str(e)}"

    elapsed_time = time.time() - start_time
    print(f"===> Zeit für Iteration {index} über Fehlertyp {tag}: {elapsed_time:.6f} Sekunden")

    # Merge results with original data and define column order
    merged_data = data.combine_first(results)
    final_column_order = original_columns + new_columns
    merged_data = merged_data[final_column_order]
    
    # Zwischenergebnisse speichern
    interim_path = output_dir / f"{model_tag}_error_types_interim.csv"
    merged_data.to_csv(interim_path, index=False)
    print(f"===> Zwischenergebnisse gespeichert: {interim_path}")
    print(f"===> Aktuelle Spaltenreihenfolge: {list(merged_data.columns)}")

# Final results mit definierter Spaltenreihenfolge
final_path = output_dir / f"{model_tag}_error_types_final.csv"
merged_data.to_csv(final_path, index=False)
print(f"===> Finale Ergebnisdatei gespeichert: {final_path}")

# Print summary mit Spalteninformation
print("\n=== Zusammenfassung ===")
print(f"Verarbeitete Zeilen/Segmente: {len(data_subset)}")
print(f"Anzahl Spalten: {len(merged_data.columns)}")
#print("\nSpaltenreihenfolge:")
#for i, col in enumerate(merged_data.columns, 1):
#    print(f"{i}. {col}")
print("\nErste Zeilen der Ergebnisse:")
print(merged_data.head())

# Summierte Wahrscheinlichkeiten der Fehlertypen berechnen und ausgeben
print("\nSummierte Wahrscheinlichkeiten der Fehlertypen:")
prob_columns = [col for col in merged_data.columns if col.endswith(' p')]
for col in prob_columns:
    error_tag = col.split(' p')[0]
    total_prob = merged_data[col].sum()
    mean_prob = merged_data[col].mean()
    print(f"{error_tag}: Summe = {total_prob:.2f}, Mittelwert = {mean_prob:.4f}")

# Fehlertypen nach Gesamtwahrscheinlichkeit sortieren
error_summary = []
for col in prob_columns:
    error_tag = col.split(' p')[0]
    total_prob = merged_data[col].sum()
    mean_prob = merged_data[col].mean()
    error_summary.append({'tag': error_tag, 'summe': total_prob, 'mittelwert': mean_prob})

error_df = pd.DataFrame(error_summary)
error_df = error_df.sort_values('summe', ascending=False)
print("\nFehlertypen nach Gesamtwahrscheinlichkeit (absteigend):")
print(error_df)