import json
import ollama


def read_combined_text(file_path: str) -> str:
    """
    Reads the content of the combined.txt file.

    :param file_path: Path to the combined.txt file.
    :return: Content of the file as a string.
    """
    with open(file_path, "r") as file:
        return file.read()


def query_llama_model(prompt: str) -> str:
    """
    Queries the LLaMA model using Ollama's Python library and returns the output.

    :param prompt: The input prompt for the LLaMA model.
    :return: Model output as a string.
    """
    # Query the LLaMA model
    stream = ollama.chat(
        model='mistral-nemo:12b-instruct-2407-q3_K_S',  # 'llama3.1:8b',
        messages=[
            {'role': 'user', 'content': prompt}
        ],
        stream=True,
        # format='json'
    )

    # Collect output chunks
    model_output = ""
    for chunk in stream:
        model_output += chunk['message']['content']

    return model_output


def extract_information_from_text(file_path: str):
    """
    Extracts artist, producer, and editor information from the combined.txt file
    and saves it as a JSON file.

    :param file_path: Path to the combined.txt file.
    """
    # Read the content from the combined.txt file
    combined_text = read_combined_text(file_path)

    # Define the prompt for the LLaMA model
    PROMPT = f'''[INST] 
    You are given the extracted text from the OCR process of a vinyl record's cover. The OCR text may contain errors such as cut-off words, missing characters, or incorrect formatting. Your task is to identify and extract specific pieces of information from this text, correcting these errors where possible, and present the information in a structured JSON format. The output should include general information about the vinyl (LP), details about each track, and any relevant notes.

    Instructions:

    1. General Information:
    - LP_ID: Extract the ID of the LP, which always starts with "LP" followed by a 4-digit number.
    - Title: Extract the title of the LP, typically found on the front cover.
    - Subtitle: If present, extract the subtitle of the LP.
    - Performer: Extract the full names of the performers or artists involved, separated by a semicolon if multiple.
    - Publisher: Extract the name of the publisher (company or organization responsible for the release).
    - Publishing Year: Identify the publishing year of the LP if mentioned.
    - Label Company: Extract the name of the label company.
    - Label Number: Identify the unique label identifier, usually appearing after the label company's name.
    - Language: Determine the language used in the text (e.g., Spanish).
    - Recording Info: Extract information about where or by whom the LP was recorded, if available.
    - Other Information: Include any other relevant information that doesn't fall into the above categories.
    - Genre/Style: Determine the genre or style of the album if mentioned, sometimes found in parentheses '()' after the track names.
    - Notes: Extract any additional notes or contextual information found on the cover that may provide insights into the album's content or production.

    2. Track Information:
    For each track, extract the following details:
    - Face: Determine whether the track is on Face "A" or "B".
    - Track_Number: Correctly identify the track's number or position in the list. If the track numbers are concatenated or split incorrectly, infer the correct order.
    - Track_Name: Correctly extract the name of the track. If parts of the name are cut off or incorrectly joined with other text, attempt to infer the full name.
    - Track_Composer: Extract the composer’s name, which may follow the track name (e.g., Jose Fernandez, Dpto. del Folklore).
    - Track_Length: Extract the track length if mentioned.

    3. Error Handling:
    - When you encounter cut-off words or concatenated track names/numbers, make an educated guess to reconstruct the proper names and numbers.
    - If the track numbers are listed in a format such as "1, 2, TrackName1, TrackName2", correctly separate and assign the track numbers to the respective track names.

    Example Output in JSON format:

    {{
        "General Information": {{
            "LP_ID": "LP 2837",
            "Title": "Fiesta en Bolivia",
            "Subtitle": "Duo los romanceros",
            "Performer": "Juan Espinoza; Gregorio Pinto",
            "Publisher": "Casa Alvarez",
            "Publishing Year": "1990",
            "Label Company": "Discos Alvarez",
            "Label Number": "MAS-3001",
            "Language": "Spanish",
            "Recording Info": "Grabado en estudios 'Electro Disc.'",
            "Other Information": "NaN",
            "Genre/Style": "Bolivian cueca",
            "Notes": "Original de la portada gentileza de foto Reflex Cochabamba Bolivia"
        }},
        "Track Info": [
            {{
                "Face": "A",
                "Track_Number": 1,
                "Track_Name": "Huerfana Virginia",
                "Track_Composer": "Dpto. Del Folk",
                "Track_Length": "2'44"
            }}
            // There should be 12 tracks in total
        ]
    }}

    Use this format to structure the extracted information from the following OCR text, correcting errors as necessary: 
    {combined_text}
    [/INST]
    '''

    # Query the LLaMA model
    model_output = query_llama_model(prompt=PROMPT)

    # Check if model output is valid
    if model_output.strip():
        # Parse the model output to JSON
        try:
            extracted_info = json.loads(model_output)
            output_file_path = "extracted_info.json"

            # Save the extracted information to a JSON file
            with open(output_file_path, "w") as json_file:
                json.dump(extracted_info, json_file, indent=4)

            print(
                f"Information successfully extracted and saved to {output_file_path}")

        except json.JSONDecodeError:
            print("Failed to parse model output to JSON format. Here is the raw output:")
            print(model_output)
    else:
        print("No output received from the model.")


if __name__ == "__main__":
    # Path to the combined.txt file
    combined_text_path = "/Users/lucazosso/Desktop/Luca_Sandbox_Env/DATA_MEG_PROJ/OCR-AI-Vinyl-Classification/ocr-meg-collection/ds_pipeline/0_raw_ocr_txt/LP2836_combined.txt"

    # Extract information from text
    extract_information_from_text(combined_text_path)
