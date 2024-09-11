import os
import logging

from google.cloud import vision
from dotenv import load_dotenv
# Import the progress bar decorator

# Load environment variables from the .env file
load_dotenv()

# Fetch the path to the JSON credentials from the environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS")

print("Loaded Successfully")


def extract_text_from_image(image_path: str) -> str:
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
        # Initialize the Vision API client
        client = vision.ImageAnnotatorClient()

        # Read the image file
        with open(image_path, "rb") as image_file:
            content = image_file.read()

        # Create an Image object
        image = vision.Image(content=content)

        # Specify image context language
        image_context = vision.ImageContext(language_hints=["es", "en"])

        # Perform text detection
        response = client.text_detection(
            image=image, image_context=image_context)
        texts = response.text_annotations

        # Check for errors in the response
        if response.error.message:
            raise Exception(
                f"{response.error.message}\nFor more info on error messages, "
                "check: https://cloud.google.com/apis/design/errors"
            )

        # Keep only the first detected text
        if texts:
            first_detected_text = texts[0].description.strip()
            print(f'First detected text: "{first_detected_text}"')
            return first_detected_text
        else:
            logging.info("No text detected.")
            return ""

    except Exception as e:
        logging.error(f"Error occurred while extracting text: {e}")
        return ""


# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "/Volumes/LaCie_500G/MEG_CAT_TEST/LP2836/LP2836_cover02.jpeg"
    extracted_text = extract_text_from_image(image_path)
    print(f"Extracted text:\n{extracted_text}")
