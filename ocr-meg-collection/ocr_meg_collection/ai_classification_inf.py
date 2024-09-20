# AI-Class
import os
import logging
import json
import re
import time
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting
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

GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")

print("Google Service Account JSON File Creds Loaded Successfully for VERTEXAI INFERENCE")


class AIClassifier:
    def __init__(self, max_workers=4, sleep_interval=6):
        """
        Initializes the AIClassifier class.

        Args:
            max_workers (int): The maximum number of threads to use for concurrent processing.
            sleep_interval (int): The number of seconds to wait between API calls to avoid rate limits.
        """
        self.max_workers = max_workers
        self.sleep_interval = sleep_interval

        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Set input and target directories
        self.INPUT_DIR = os.path.join(
            script_dir, '..', 'ds_pipeline', '0_raw_ocr_txt')
        print("Input Directory for AI Classification:", self.INPUT_DIR)

        self.TARGET_DIR = os.path.join(
            script_dir, '..', 'ds_pipeline', '1_json_inf_outputs')
        print("Output Directory for AI Classification:", self.TARGET_DIR)

        # Initialize Vertex AI
        vertexai.init(project=GOOGLE_PROJECT_ID, location="us-central1")
        logging.info("Initialized Vertex AI.")

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

    def _clean_json_from_response(self, response_text: str) -> dict:
        """
        Extracts and cleans JSON data from the response text.

        :param response_text: The response text containing JSON data.
        :return: A dictionary containing the parsed JSON data.
        """
        try:
            # Attempt to find the JSON part in the response text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start == -1 or json_end == -1:
                logging.error("No JSON format found in the response.")
                return {}

            json_str = response_text[json_start:json_end]
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}")
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

            # Define the system and user prompts and generate content
            SYSTEM_PROMPT = """As an expert in document entity extraction,
            you parse txt documents to identify and organize specific entities
            from diverse sources into structured formats,
            following detailed guidelines for clarity and completeness.
            """

            USER_PROMPT = f"""
            You are a document entity extraction specialist. Given a document, your task is to extract the text value of the following entities and provide them in a structured JSON format:
            {{
                "General Information": {{
                    "LP_ID": "",  // Extract the ID of the LP, **always** starts with "LP" followed by a 4-digit number + appears only in the *Front Cover* section.
                    "Country": "",  // Extract the country
                    "Title": "",  // Extract the title of the LP, typically found on the front cover.
                    "Subtitle": "",  // If present, extract the subtitle of the LP.
                    "Performer": "",  // Extract the full names of the performers or music composer involved, separated by a semicolon if multiple.
                    "Publisher": "",  // Extract the name of the publisher (company or organization responsible for the release).
                    "Publishing Year": "",  // Identify the publishing year of the LP if mentioned.
                    "Label Company": "",  // Extract the name of the label company.
                    "Label Number": "",  // Identify the unique label identifier, usually appearing after the label company's name and always starting with 2 or 3 letters followed by numbers.
                    "Language": "",  // Determine the language used in the text (e.g., Spanish).
                    "Recording Info": "",  // Extract information about where or by whom the LP was recorded, if available.
                    "Genre/Style": "",  // Determine the music genre of the album if mentioned, sometimes found in parentheses on the track name, often Iberic/South American genres.
                    "Notes": "",  // Extract any additional notes or contextual information found on the cover that may provide insights into the album's content or production.
                    "Other Information": ""  // Include any other relevant information that doesn't fall into the above categories.
                }},
                "Track Info": [
                    {{
                        "Face": "",  // Determine whether the track is on Face "A" or "B". [example: 1.a, 1b, etc.]
                        "Track_Number": "",  // Correctly identify the track's number or position in the list. Write it as a whole number without any punctuation marks.
                        "Track_Name": "",  // Correctly extract the name of the track.
                        "Track_Composer": "",  // Extract the composer’s name, follows the track name.
                        "Track_Length": ""  // Extract the track length if mentioned.
                    }}  // In most cases, there should be 6 tracks on both Faces.
                ]
            }}

            Instructions:
            1. Extract the information from the provided OCR text, filling in the fields above.
            2. Correct any errors such as cut-off words, missing characters, or incorrect formatting.
            3. If the track numbers are concatenated or incorrectly listed, infer the correct order.
            4. Use educated guesses to reconstruct names and numbers when necessary.
            5. Use only the information provided in the text.
            6. If nothing is found respond with an empty json object

            Text to Analyze:
            """

            document = combined_text

            # Define generation configuration
            generation_config = {
                "max_output_tokens": 8192,
                "temperature": 0.7,
                "top_p": 0.95,
                "response_mime_type": "application/json",
            }
            # Define custom safety settings
            safety_settings = [
                SafetySetting(
                    category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
                ),
                SafetySetting(
                    category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
                ),
                SafetySetting(
                    category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
                ),
                SafetySetting(
                    category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
                ),
            ]

            # Instantiate the generative model
            model = GenerativeModel(
                "gemini-1.5-pro-001",
                generation_config=generation_config,
                system_instruction=SYSTEM_PROMPT,
                safety_settings=safety_settings  # No safety settings for now
            )

            # Generate content using the model
            responses = model.generate_content(
                [USER_PROMPT, document],
                # generation_config=generation_config,
                stream=True,

            )

            # Concatenate all response parts into a single string
            response_text = "".join(response.text for response in responses)

            # **Print the LLM's Raw Response Text**
            # print("LLM Raw Response Text:")
            # print(response_text)
            # logging.info("Printed the LLM's raw response text.")

            # Extract JSON from response
            json_data = self._clean_json_from_response(response_text)
            # self._extract_json_from_response(response_text)

            # Save the extracted JSON to the target directory
            if json_data:
                file_name = os.path.basename(file_path).replace(
                    "_combined.txt", "_ai_output.json")
                self._save_json(json_data, file_name)
            pbar.update(1)  # Update progress bar after saving the output

            # Sleep to respect API quota limits
            time.sleep(self.sleep_interval)
            logging.info(
                f"Sleeping for {self.sleep_interval} seconds to avoid exceeding quota.")

    def batch_generate_inferences(self):
        """
        Generates inferences for multiple text files in the input directory concurrently.
        """
        txt_files = [os.path.join(self.INPUT_DIR, f) for f in os.listdir(
            self.INPUT_DIR) if f.endswith("_combined.txt")]

        # Filter out files that have already been processed
        files_to_process = []
        for file_path in txt_files:
            output_file_name = os.path.basename(file_path).replace(
                "_combined.txt", "_ai_output.json")
            output_file_path = os.path.join(self.TARGET_DIR, output_file_name)
            if not os.path.exists(output_file_path):
                files_to_process.append(file_path)
            else:
                logging.info(f"Skipping already processed file: {file_path}")

        # Process remaining files concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(
                self.generate_inference, file_path): file_path for file_path in files_to_process}

            for future in tqdm(as_completed(future_to_file), total=len(files_to_process), desc="Batch Processing LP files", unit="file"):
                file_path = future_to_file[future]
                try:
                    future.result()
                    logging.info(f"Completed inference for {file_path}")
                except Exception as exc:
                    logging.error(f"Error processing {file_path}: {exc}")


if __name__ == "__main__":
    classifier = AIClassifier()
    classifier.batch_generate_inferences()


# Script Time-Log
# quota to 10 requests per minute, we can adjust the sleep_interval accordingly. With a new quota of 10 requests per minute, the script can send a request every 6 seconds (since
# 60seconds/10requests=6seconds/request). We can set the sleep_interval to 6 seconds to ensure that we don't exceed the quota.
