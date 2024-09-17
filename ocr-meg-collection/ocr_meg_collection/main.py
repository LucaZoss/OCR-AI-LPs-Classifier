import os
import time

from dotenv import load_dotenv

from ocr_meg_collection.ocr_pipeline import OCRPipeline
from ocr_meg_collection.ai_classification_inf import AIClassifier
from ocr_meg_collection.post_processing import Orchestrator
from ocr_meg_collection.utils import get_lp_subfolders

load_dotenv()

# Set the base directory where the LP subfolders reside, the path is in the .env file
BASE_DIR = os.getenv("BASE_DIR")

print("Base Directory PATH for LP Processing", BASE_DIR)


def run_ocr(base_dir: str, lp_selection: str = "all"):
    """
    Running the OCR Process on the LP covers.

    Args:
        base_dir (str): Base directory containing LP subfolders.
        lp_selection (str, optional): A string to specify which LPs to process.
                                      Formats:
                                        - "all": Process all LPs.
                                        - "first-X" or "X": Process the first X LPs.
                                        - "last-X": Process the last X LPs.
                                        - "start-end": Process LPs from index start to end.
                                      Defaults to "all".
    """

    # Get LPs List
    lp_subfolders_list = get_lp_subfolders(base_dir=base_dir)

    # Determine LP selection based on lp_selection argument
    if lp_selection == "all":
        lp_list = lp_subfolders_list
    elif "last-" in lp_selection:
        # Extract the number of LPs to select from the end
        count = int(lp_selection.split("-")[1])
        lp_list = lp_subfolders_list[-count:]
    elif "first-" in lp_selection or lp_selection.isdigit():
        # Handle "first-X" or just a digit to select the first X LPs
        count = int(lp_selection.split(
            "-")[1] if "first-" in lp_selection else lp_selection)
        lp_list = lp_subfolders_list[:count]
    elif "-" in lp_selection:
        # Extract the start and end range from the lp_selection string
        start, end = map(int, lp_selection.split("-"))
        # Adjust for 0-based indexing
        lp_list = lp_subfolders_list[start-1:end]
    else:
        raise ValueError(
            "Invalid lp_selection format. Use 'all', 'first-X', 'X', 'last-X', or 'start-end'.")

    # Initialize the OCR pipeline
    ocr_pipeline = OCRPipeline(base_dir, lp_list)

    # Process OCR on LP covers
    start_time = time.time()
    print("Starting OCR processing...")
    ocr_pipeline.process_ocr()
    end_time = time.time()
    print(f"OCR processing completed in {end_time - start_time:.2f} seconds.")


def run_ai_classification_inference():
    """
    This function runs the AI Classification Inference on the OCR Text.
    """
    classifier = AIClassifier()

    start_time = time.time()
    print("Starting AI classification inference...")
    classifier.batch_generate_inferences()
    end_time = time.time()
    print(
        f"AI classification inference completed in {end_time - start_time:.2f} seconds.")


def run_post_processing():
    """
    Run the post-processing step to merge the JSON files into 2 different CSV files.
    """
    orchestrator = Orchestrator(lp_base_dir=BASE_DIR)

    start_time = time.time()
    print("Starting post-processing...")
    orchestrator.run()
    end_time = time.time()
    print(f"Post-processing completed in {end_time - start_time:.2f} seconds.")


def main(lp_selection: str = "all"):
    global_start_time = time.time()

    run_ocr(base_dir=BASE_DIR, lp_selection=lp_selection)
    run_ai_classification_inference()
    run_post_processing()

    global_end_time = time.time()
    print(
        f"Total execution time: {global_end_time - global_start_time:.2f} seconds.")


if __name__ == "__main__":
    # Example: Change "all" to "5", "first-7", "last-20", "21-56" as needed
    main(lp_selection="25")
