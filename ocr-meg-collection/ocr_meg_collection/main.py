import os
import time
import re
from dotenv import load_dotenv
from ocr_meg_collection.ocr_pipeline import OCRPipeline
from ocr_meg_collection.ai_classification_inf import AIClassifier
from ocr_meg_collection.post_processing import Orchestrator
from ocr_meg_collection.utils import get_lp_subfolders
from ocr_meg_collection.cleaner import Cleaner

# Load environment variables
load_dotenv()

# Set the working directory to the script's location
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Current working directory:", os.getcwd())

# Set the base directory where the LP subfolders reside, the path is in the .env file
BASE_DIR = os.getenv("BASE_DIR")

print("Base Directory PATH for LP Processing:", BASE_DIR)


def natural_sort(lp_list):
    """
    Sorts the list of LP subfolders in natural order.

    Args:
        lp_list (list): List of LP subfolders.

    Returns:
        list: Sorted list of LP subfolders.
    """
    return sorted(lp_list, key=lambda x: int(re.search(r'\d+', x).group()))


def get_lp_subfolders(base_dir):
    """
    Retrieves the LP subfolders from the specified base directory.

    Args:
        base_dir (str): The base directory where LP subfolders are located.

    Returns:
        list: Sorted list of LP subfolder names.
    """
    print(f"Base directory: {base_dir}")
    try:
        subfolders = [f.name for f in os.scandir(
            base_dir) if f.is_dir() and f.name.startswith("LP")]
        print(f"Filtered subfolder names: {subfolders}")
        sorted_subfolders = natural_sort(subfolders)
        print(f"Sorted subfolder names: {sorted_subfolders}")
        return sorted_subfolders
    except Exception as e:
        print(f"Error while listing subfolders: {e}")
        return []


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
    # Change to base directory and list subdirectories
    original_dir = os.getcwd()  # Store original directory
    os.chdir(base_dir)
    lp_subfolders_list = [d for d in os.listdir(
    ) if os.path.isdir(d) and d.startswith('LP')]

    # Sort LP subfolders in natural order
    lp_subfolders_list = natural_sort(lp_subfolders_list)

    # Determine LP selection based on lp_selection argument
    if lp_selection == "all":
        lp_list = lp_subfolders_list

    elif "last-" in lp_selection:
        count = int(lp_selection.split("-")[1])
        lp_list = lp_subfolders_list[-count:]

    elif "first-" in lp_selection or lp_selection.isdigit():
        count = int(lp_selection.split(
            "-")[1] if "first-" in lp_selection else lp_selection)
        lp_list = lp_subfolders_list[:count]
    elif "-" in lp_selection:
        start, end = map(int, lp_selection.split("-"))
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

    # Revert back to the original directory
    os.chdir(original_dir)


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


def run_cleaner():
    """
    Run Cleaner to clean and label the general and track info.
    """
    raw_csv_input = os.path.join(
        os.getcwd(), 'ocr-meg-collection', 'ds_pipeline', '2_raw_csv')
    clean_csv_output = os.path.join(
        os.getcwd(), 'ocr-meg-collection', 'ds_pipeline', '3_clean_csv')

    cleaner_instance = Cleaner(
        input_dir=raw_csv_input, output_dir=clean_csv_output)
    start_time = time.time()
    print("Starting cleaning...")
    cleaner_instance.run_cleaner()
    end_time = time.time()
    print(f"Cleaning completed in {end_time - start_time:.2f} seconds.")


def main(lp_selection: str = "all"):
    global_start_time = time.time()

    run_ocr(base_dir=BASE_DIR, lp_selection=lp_selection)
    run_ai_classification_inference()
    run_post_processing()
    run_cleaner()

    global_end_time = time.time()
    print(
        f"Total execution time: {global_end_time - global_start_time:.2f} seconds.")


if __name__ == "__main__":
    # Only print the list of LPs
    # print(get_lp_subfolders(base_dir=BASE_DIR))
    # Example: Change "all" to "5", "first-7", "last-20", "21-56" as needed
    main(lp_selection="38-38")
