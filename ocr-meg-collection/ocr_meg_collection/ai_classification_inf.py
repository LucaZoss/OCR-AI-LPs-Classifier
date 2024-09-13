import os
import logging
import json
import re
import vertexai
from vertexai.generative_models import GenerativeModel
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Import tqdm for progress bars

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from the .env file
load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS")

print("Google Service Account JSON File Creds Loaded Successfully for VERTEXAI INFERENCE")


class AIClassifier:
    def __init__(self, max_workers=4):
        """
        Initializes the AIClassifier class.

        Args:
            max_workers (int): The maximum number of threads to use for concurrent processing.
        """
        self.max_workers = max_workers

        # Set input and target directories
        self.INPUT_DIR = os.path.join(os.getcwd(),
                                      'ocr-meg-collection', 'ds_pipeline', '0_raw_ocr_txt')
        self.TARGET_DIR = os.path.join(os.getcwd(),
                                       'ocr-meg-collection', 'ds_pipeline', '1_json_inf_outputs')

    def _read_input_txt_file(self, file_path: str) -> str:
        """
        Reads the content of the combined.txt file.

        :param file_path: Path to the combined.txt file.
        :return: Content of the file as a string.
        """
        logging.info(f"Reading file: {file_path}")
        with open(file_path, "r") as file:
            return file.read()

    def _save_json(self, data: dict, file_name: str):
        """
        Saves the JSON data to a file in the target directory.

        :param data: JSON data to save.
        :param file_name: Name of the file to save the data.
        """
        target_file_path = os.path.join(self.TARGET_DIR, file_name)
        with open(target_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        logging.info(f"Saved JSON output to {target_file_path}")

    def _extract_json_from_response(self, response_text: str) -> dict:
        """
        Extracts JSON from the LLM's response text formatted in markdown.

        :param response_text: Response text from the LLM.
        :return: Extracted JSON data as a dictionary.
        """
        logging.info("Extracting JSON from the LLM response.")
        json_match = re.search(
            r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to decode JSON: {e}")
        else:
            logging.error("No JSON format found in the response.")
        return {}

    def generate_inference(self, file_path: str):
        """
        Generates inferences from the combined OCR text using Vertex AI's Generative Model.

        :param file_path: Path to the combined OCR text file.
        """
        with tqdm(total=3, desc=f"Processing {os.path.basename(file_path)}", unit="step") as pbar:
            # Step 1: Read the combined text from the specified file path
            combined_text = self._read_input_txt_file(file_path)
            pbar.update(1)  # Update progress bar after reading the file

            # Step 2: Initialize Vertex AI
            vertexai.init(project="lz-test-350609", location="us-central1")
            logging.info("Initialized Vertex AI.")
            pbar.update(1)  # Update progress bar after initializing Vertex AI

            # Step 3: Define the system and user prompts and generate content
            # Define the system and user prompts
            SYSTEM_PROMPT = """As an expert in document entity extraction,
            you parse txt documents to identify and organize specific entities
            from diverse sources into structured formats,
            following detailed guidelines for clarity and completeness.
            """

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

        # Concatenate all response parts into a single string
        response_text = "".join(response.text for response in responses)

        # Extract JSON from response
        json_data = self._extract_json_from_response(response_text)

        # Save the extracted JSON to the target directory
        if json_data:
            file_name = os.path.basename(file_path).replace(
                "_combined.txt", "_ai_output.json")
            self._save_json(json_data, file_name)
        pbar.update(1)  # Update progress bar after saving the output

    def batch_generate_inferences(self):
        """
        Generates inferences for multiple text files in the input directory concurrently.
        """
        txt_files = [os.path.join(self.INPUT_DIR, f) for f in os.listdir(
            self.INPUT_DIR) if f.endswith("_combined.txt")]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(
                self.generate_inference, file_path): file_path for file_path in txt_files}

            for future in tqdm(as_completed(future_to_file), total=len(txt_files), desc="Batch Processing LP files", unit="file"):
                file_path = future_to_file[future]
                try:
                    future.result()
                    logging.info(f"Completed inference for {file_path}")
                except Exception as exc:
                    logging.error(f"Error processing {file_path}: {exc}")


if __name__ == "__main__":
    classifier = AIClassifier()
    classifier.batch_generate_inferences()
