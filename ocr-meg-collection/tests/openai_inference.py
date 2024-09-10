from openai import OpenAI
import json

import pandas as pd


def read_combined_text(file_path: str) -> str:
    """
    Reads the content of the combined.txt file.

    :param file_path: Path to the combined.txt file.
    :return: Content of the file as a string.
    """
    with open(file_path, "r") as file:
        return file.read()


def query_gpt4o(prompt: str):
    api_key = "sk-7mFl0kHX0-EheoipR2JzGLEnPTFsiACZ_t1O_fd3TpT3BlbkFJk0uEfZNLfvwTPYq6NkLet7Q_vvz43SB9FRsZer7vYA"

    client = OpenAI(
        # This is the default and can be omitted
        api_key=api_key,
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o",

    )

    return chat_completion.choices[0].message.content


def extract_information_from_text(file_path: str):
    """
    Extracts artist, producer, and editor information from the combined.txt file
    and saves it as a JSON file.

    :param file_path: Path to the combined.txt file.
    """
    # Read the content from the combined.txt file
    combined_text = read_combined_text(file_path)

    # Define the prompt for the LLaMA model
    PROMPT = f'''   
    You are provided with text extracted from the front and back covers of a vinyl record. Your task is to identify and extract specific information from each section as follows:
    
    ### Information to Extract from the Front Cover:
    1. **Title**: The main title of the album, as it appears on the front side of the cover.
    2. **Subtitle**: Any subtitle of the album, also found on the front side of the cover.
    3. **Interpreter/Artist**: The main artists, authors, or interpreters associated with the album.
    4. **Producer**: The name of the producer, which could be an organization or an individual.
    5. **Publisher**: The name of the publisher, usually a corporate entity.
    6. **Publisher ID**: Any identifier associated with the publisher, such as a code (e.g., "H-1234").

    ### Information to Extract from the Back Cover:
    1. **Track Names**: The names of all tracks listed on both sides (Face A and Face B) of the vinyl.
    - Tracks often start with numbers or markers such as "1. Track 1," "2. Track 2," etc.
    - Tracks may be separated by Face A and Face B.
    2. **Track Lengths**: The length or duration of each track.
    
    ### Output Format: Write a properly formatted JSON file with the extracted information.

    ### Guidelines:
    - Only Use Provided Information: Extract information strictly from the provided text sections. Do not infer or create information beyond what is available.
    - Maintain Formatting: Ensure that extracted names, titles, and other information are copied exactly as they appear in the text.
    - If specific information is not available in the text, mark the field with "NaN".
    
    text: {combined_text}
    '''

    # Query the LLaMA model
    model_output = query_gpt4o(prompt=PROMPT)

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

    return model_output


if __name__ == "__main__":
    # Path to the combined.txt file
    combined_text_path = "/Users/lucazosso/Desktop/Luca_Sandbox_Env/DATA_MEG_PROJ/OCR-AI-Vinyl-Classification/ds/extracted_text/20240906_131209_combined.txt"

    # Extract information from text
    model_output = extract_information_from_text(combined_text_path)

    # Delete "```json" from the beginning of the model output and "```" from the end
    model_output = model_output.replace("```json", "").replace("```", "")

    # Store model output as a txt file
    with open("model_output.json", "w") as text_file:
        text_file.write(model_output)
