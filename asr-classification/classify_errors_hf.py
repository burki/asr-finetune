import pandas as pd
from langchain.prompts import PromptTemplate
import re
from transformers import pipeline
import ast
import time


def normalize(text):
    """
    Removes certain characters from text and converts to lowercase.

    Args:
        text (str or list of str): Single string or list of strings to be normalized.

    Returns:
        str or list of str: Normalized string or list of normalized strings.
    """

    def process_single_text(single_text):
        result = single_text.strip().lower()
        result = re.sub(r"[!?.,\]\[\\;']", "", result)
        return result

    if isinstance(text, list):
        return [process_single_text(t) for t in text]
    elif isinstance(text, str):
        return process_single_text(text)
    else:
        raise TypeError("Input must be a string or a list of strings.")


# Load dataset (CSV or JSON fallback)
finetuned = False
model_tag = 'small_eg_fzh_combined_v2_transformers' if finetuned else 'small_eg_fzh_combined_v2_whisper'

try:
    data = pd.read_csv(model_tag + '_error_types_0.csv')
except:
    data = pd.read_json(model_tag + '.json')

df = pd.read_csv('error_types_new.csv')  # Load error types metadata

# Load the Meta LLaMA model for text generation
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
nlp_pipeline = pipeline("text-generation", model=model_id, device="cuda",
                        model_kwargs={"torch_dtype": torch.bfloat16},
                        max_new_tokens=1000,
                        temperature=0.7)

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# Define structured response format
class SlideFormatter(BaseModel):
    probability: float = Field(description="Probability of error occurrence")


def classify_error_type(row, error_type, error_description, example, finetuned=False):
    """
    Classifies errors in transcription output using a language model.

    Args:
        row (pd.Series): A row containing original and predicted text.
        error_type (str): The type of error being classified.
        error_description (str): Detailed description of the error type.
        example (str): Example of the error.
        finetuned (bool): Flag to determine which model to evaluate

    Returns:
        tuple: (probability of error, occurrence count, explanation)
    """
    original = normalize(row['original']) if finetuned else normalize(row['target'])
    prediction = normalize(row['predictions']) if finetuned else normalize(row['actual'])

    format_instructions = """
    Important: Respond with only the probability as a number. 
    Do not include any text, explanation, or additional output.
    """

    # Define system prompt
    prompt_system = (
        "Du erhältst einen Originaltext, welcher einen Ausschnitt eines Interviews "
        "entspricht, und eine Transkription eines Audio-zu-Text (kurz: ASR) Modelles. "
        "Prüfe, ob der Fehlertyp in der Transkription enthalten ist. "
        "Falls du dir 100% sicher bist, dass der Fehlertyp aufgetreten ist, gebe eine Wahrscheinlichkeit von 1 aus. "
        "Falls du dir 100% sicher bist, dass der Fehlertyp nicht aufgetreten ist, gebe eine Wahrscheinlichkeit von 0 aus. "
        "Verwende 0 oder 1 nur im Falle von absoluter Sicherheit. "
        "Bei Unsicherheit verwende eine Zahl zwischen 0 und 1. "
        "Deine Antwort sollte als JSON-Objekt formatiert sein: "
        "{\"p\": \"<Wahrscheinlichkeit>\", \"N\": \"<Fehlertyp-Anzahl>\", \"warum\": \"<Begründung>\"}""
    )

    prompt = f"""Originaltext: {original} \nTranskription: {prediction} \nFehlertyp: {error_type} \nBeschreibung: {error_description} \nBeispiel: {example}"""

    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt}
    ]

    # Run model inference
    response = nlp_pipeline(messages)

    try:
        answer = ast.literal_eval(response[0]["generated_text"][-1]["content"])
        probability = float(answer["p"])
        N_error = int(answer["N"])
        explain = answer["warum"]
    except Exception as e:
        print(f"Error parsing LLM response for error type '{error_type}': {response}")
        probability, N_error, explain = 0.0, 0, 'fehler'

    return probability, N_error, explain


# Apply classification for first 100 rows
for index, df_row in df.head(100).iterrows():
    start_time = time.time()

    error_type = df_row['Fehlertyp']
    description = df_row['Definition']
    example = df_row['Beispiel']
    tag = df_row['tag']

    prob_col = f"{tag} p"
    count_col = f"{tag} N"
    warum_col = f"{tag} warum"

    data[[prob_col, count_col, warum_col]] = data.apply(
        lambda row: pd.Series(classify_error_type(row, error_type, description, example, finetuned=finetuned)), axis=1)

    elapsed_time = time.time() - start_time
    print(f"Time taken for iteration: {elapsed_time:.6f} seconds")

    # Save intermediate results
    data.to_csv(model_tag + '_error_types_0.csv', index=False)

# Save final results
data.to_csv(model_tag + '_error_types.csv', index=False)
