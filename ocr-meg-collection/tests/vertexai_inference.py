import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting, FinishReason
import dotenv
import os

# Load environment variables from .env file
dotenv.load_dotenv()

# Login with service account JSON credentials file (path stored in .env file)
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")


def read_combined_text(file_path: str) -> str:
    """
    Reads the content of the combined.txt file.

    :param file_path: Path to the combined.txt file.
    :return: Content of the file as a string.
    """
    with open(file_path, "r") as file:
        return file.read()


def generate(file_path: str):
    # Read the combined text from the specified file path
    combined_text = read_combined_text(file_path)

    # Initialize Vertex AI
    vertexai.init(project="lz-test-350609", location="us-central1")

    # Define the system and user prompts
    SYSTEM_PROMPT = """As an expert in document entity extraction, you parse txt documents to identify and organize specific entities from diverse sources into structured formats, following detailed guidelines for clarity and completeness."""

    USER_PROMPT = """You are a document entity extraction specialist. Given a document, your task is to extract the text value of the following entities and provide them in a structured JSON format:
    { \"General Information\": 
        { 
            \"LP_ID\": \"\", // Extract the ID of the LP, always starts with \"LP\" followed by a 4-digit number. 
            \"Title\": \"\", // Extract the title of the LP, typically found on the front cover. 
            \"Subtitle\": \"\", // If present, extract the subtitle of the LP. 
            \"Performer\": \"\", // Extract the full names of the performers or music composer involved, separated by a semicolon if multiple. 
            \"Publisher\": \"\", // Extract the name of the publisher (company or organization responsible for the release). 
            \"Publishing Year\": \"\", // Identify the publishing year of the LP if mentioned. 
            \"Label Company\": \"\", // Extract the name of the label company. 
            \"Label Number\": \"\", // Identify the unique label identifier, usually appearing after the label company\'s name. 
            \"Language\": \"\", // Determine the language used in the text (e.g., Spanish). 
            \"Recording Info\": \"\", // Extract information about where or by whom the LP was recorded, if available.  
            \"Genre/Style\": \"\", // Determine the music genre of the album if mentioned, sometimes found in parentheses on the track name  often iberic/south american genres
            \"Notes\": \"\" // Extract any additional notes or contextual information found on the cover that may provide insights into the album\'s content or production. 
            \"Other Information\": \"\", // Include any other relevant information that doesn\'t fall into the above categories.
        }, 
    {\"Track Info\": [ 
        { 
            \"Face\": \"\", // Determine whether the track is on Face \"A\" or \"B\". 
            \"Track_Number\": \"\", // Correctly identify the track\'s number or position in the list. 
            \"Track_Name\": \"\", // Correctly extract the name of the track. 
            \"Track_Composer\": \"\", // Extract the composer’s name, which may follow the track name. 
            \"Track_Length\": \"\" // Extract the track length if mentioned. 
    } // In most cases, there should be 6 tracks on both Face] }

    Instructions: 1. Extract the information from the provided OCR text, filling in the fields above. 2. Correct any errors such as cut-off words, missing characters, or incorrect formatting. 3. If the track numbers are concatenated or incorrectly listed, infer the correct order. 4. Use educated guesses to reconstruct names and numbers when necessary. 

    Text to Analyze:"""

    document = combined_text

    # Define generation configuration
    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 0.7,
        "top_p": 0.95,
    }

    # Instantiate the generative model
    model = GenerativeModel(
        "gemini-1.5-pro-001",
        system_instruction=SYSTEM_PROMPT
    )

    # Generate content using the model
    responses = model.generate_content(
        [USER_PROMPT, document],
        generation_config=generation_config,
        stream=True,
    )

    # Print the responses
    for response in responses:
        print(response.text, end="")


if __name__ == "__main__":
    # Replace with your combined.txt file path
    file_path = "/Users/lucazosso/Desktop/Luca_Sandbox_Env/DATA_MEG_PROJ/OCR-AI-Vinyl-Classification/ocr-meg-collection/ds_pipeline/0_raw_ocr_txt/LP2836_combined.txt"
    generate(file_path=file_path)
