import os
import logging
from ocr_meg_collection.utils import fetch_lp_covers_path
from google.cloud import vision
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from the .env file
load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS")

print("Google Service Account JSON File Creds Loaded Successfully")

TARGET_DIR = os.path.join(os.getcwd(),
                          'ocr-meg-collection', 'ds_pipeline', '0_raw_ocr_txt')


class OCRPipeline:
    def __init__(self, base_dir: str, lp_list: list, max_workers=4):
        """
        Initializes the OCRPipeline class.

        Args:
            base_dir (str): Path to the directory containing the numerical folders of each LP.
            lp_list (list): List of the LPs to be processed by OCR.
            max_workers (int): The maximum number of threads to use for concurrent processing.
        """
        self.base_dir = base_dir
        self.lp_list = lp_list
        self.cover_paths = self._get_all_lp_covers_path()
        self.max_workers = max_workers

    def _get_all_lp_covers_path(self) -> dict:
        """
        Retrieves the paths of all LP covers for each LP in the list.

        Uses the fetch_lp_covers_path utility function to obtain the front and back cover paths
        for each LP specified in the lp_list.

        Returns:
            dict: A dictionary where each key is an LP identifier and the value is a list containing the paths
                  to the front and back covers of the LP.
        """
        covers_path_dict = {}
        for lp in tqdm(self.lp_list, desc="Fetching LP cover paths", unit="LP"):
            try:
                cover_paths = fetch_lp_covers_path(self.base_dir, lp)
                covers_path_dict[lp] = cover_paths
                logging.info(f"Fetched paths for LP: {lp}")
            except FileNotFoundError:
                logging.error(
                    f"LP '{lp}' does not exist in the specified directory: {self.base_dir}")
        return covers_path_dict

    def process_ocr(self):
        """
        Processes OCR on the LP covers and stores the extracted text in the target directory.
        """
        logging.info("Starting OCR process...")

        # Use a ThreadPoolExecutor to handle multiple images concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_album = {
                executor.submit(self._process_single_album, album, paths): album
                for album, paths in self.cover_paths.items()
            }

            # Use tqdm to display progress
            for future in tqdm(as_completed(future_to_album), total=len(future_to_album), desc="Processing OCR for LPs", unit="LP"):
                album = future_to_album[future]
                try:
                    future.result()  # Get the result to check for any raised exceptions
                except Exception as e:
                    logging.error(f"Error processing LP '{album}': {e}")

        logging.info("OCR process completed.")

    def _process_single_album(self, album, paths):
        """
        Processes a single album by extracting text from its front and back covers.

        Args:
            album (str): The album identifier.
            paths (list): A list containing paths to the front and back covers.
        """
        if not paths or len(paths) != 2:
            logging.warning(
                f"Skipping LP '{album}' due to missing cover images.")
            return

        # Extract front and back image paths
        front_image_path, back_image_path = paths

        # Check if front and back images are not None and exist
        if not front_image_path or not back_image_path:
            logging.warning(
                f"Skipping LP '{album}' due to one or both image paths being None.")
            return

        if not all(os.path.exists(p) for p in [front_image_path, back_image_path]):
            logging.warning(
                f"Skipping LP '{album}' due to missing image files.")
            return

        # Extract text from front and back covers
        front_text = self._extract_text_from_image(front_image_path)
        back_text = self._extract_text_from_image(back_image_path)

        # Store the extracted text in a single file
        self._store_text_to_file(album, front_text, back_text)

    def _extract_text_from_image(self, image_path: str) -> str:
        """
        Extracts only the first detected text from the provided image path using Google Cloud Vision API.

        Args:
            image_path (str): Path to the image from which to extract text.

        Returns:
            str: The first detected text as a single string.
        """
        if not image_path or not os.path.exists(image_path):
            logging.error(f"Image file '{image_path}' does not exist.")
            return ""

        logging.info(f"Extracting text from image: {image_path}")

        try:
            client = vision.ImageAnnotatorClient()

            with open(image_path, "rb") as image_file:
                content = image_file.read()

            image = vision.Image(content=content)
            image_context = vision.ImageContext(language_hints=["es", "en"])

            response = client.text_detection(
                image=image, image_context=image_context)
            texts = response.text_annotations

            if response.error.message:
                raise Exception(
                    f"{response.error.message}\nFor more info on error messages, "
                    "check: https://cloud.google.com/apis/design/errors"
                )

            if texts:
                first_detected_text = texts[0].description.strip()
                return first_detected_text
            else:
                logging.info("No text detected.")
                return ""

        except Exception as e:
            logging.error(f"Error occurred while extracting text: {e}")
            return ""

    def _store_text_to_file(self, album_name: str, front_text: str, back_text: str):
        """
        Stores the extracted text from both front and back covers into a single text file.

        Args:
            album_name (str): The name of the LP album.
            front_text (str): Extracted text from the front cover.
            back_text (str): Extracted text from the back cover.
        """
        os.makedirs(TARGET_DIR, exist_ok=True)

        file_path = os.path.join(TARGET_DIR, f"{album_name}_combined.txt")
        with open(file_path, "w") as file:
            file.write("Front Cover:\n")
            file.write(front_text)
            file.write("\nBack Cover:\n")
            file.write(back_text)
        logging.info(f"Text from album cover faces stored in {file_path}")


if __name__ == "__main__":
    base_dir = '/Volumes/LaCie_500G/MEG_CAT_TEST'
    lp_list = ['LP2836', 'LP2837']
    ocr_pipeline = OCRPipeline(
        base_dir=base_dir, lp_list=lp_list, max_workers=4)

    ocr_pipeline.process_ocr()
