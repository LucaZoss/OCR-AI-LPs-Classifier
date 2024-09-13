import os
import time

from dotenv import load_dotenv

from ocr_meg_collection.ocr_pipeline import OCRPipeline
from ocr_meg_collection.ai_classification_inf import AIClassifier
from ocr_meg_collection.post_processing import Orchestrator
from ocr_meg_collection.utils import get_lp_subfolders

# load_dotenv()

# Set the base directory where the LP subfolders reside, the path is in the .env file
BASE_DIR = os.getenv("BASE_DIR")

print("Base Directory PATH for LP Processing", BASE_DIR)


def run_ocr(base_dir: str, lps_to_process: int = all):
    """_summary_
    Running the OCR Process on the LP covers.
    base_dir (str): _description_
    lps_to_process (int, optional): _description_. Defaults to all.
    """

    # Get LPs List
    lp_subfolders_list = get_lp_subfolders(base_dir=base_dir)

    if lps_to_process != all:
        lp_list = lp_subfolders_list[:lps_to_process]
    else:
        lp_list = lp_subfolders_list

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
    This function run the AI Classification Inference on the OCR Text.
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
    Run the post-processing step to merge the JSON files into 2 different csv files.
    """
    orchestrator = Orchestrator(lp_base_dir=BASE_DIR)

    start_time = time.time()
    print("Starting post-processing...")
    orchestrator.run()
    end_time = time.time()
    print(f"Post-processing completed in {end_time - start_time:.2f} seconds.")


def main(lps_to_process: int = all):
    global_start_time = time.time()

    run_ocr(base_dir=BASE_DIR, lps_to_process=lps_to_process)
    run_ai_classification_inference()
    run_post_processing()

    global_end_time = time.time()
    print(
        f"Total execution time: {global_end_time - global_start_time:.2f} seconds.")


if __name__ == "__main__":
    main(lps_to_process=1)
