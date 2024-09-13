import os
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Get the current working directory
current_dir = os.getcwd()
# Go back one directory
parent_dir = os.path.dirname(current_dir)

TARGET_DIR = os.path.join(parent_dir, 'ds', 'extracted_text')

# Function to extract text from image using Tesseract with pre/post-processing


def extract_text_from_img(image_path: str) -> str:
    """
    Extracts text from the provided image path using Tesseract OCR.

    :param image_path: Path to the image from which to extract text.
    :return: Extracted text as a single string.
    """
    # Read the image from which text needs to be extracted
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Performing OTSU threshold
    _, thresh1 = cv2.threshold(
        gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # Dilation parameter, bigger means less rect
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))

    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

    # Finding contours
    contours, _ = cv2.findContours(
        dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Creating a copy of the image
    im2 = gray.copy()

    cnt_list = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Cropping the text block for OCR input
        cropped = im2[y:y + h, x:x + w]

        # Apply OCR on the cropped image
        text = pytesseract.image_to_string(cropped)

        cnt_list.append([x, y, text])

    # Sort text by their y-coordinates to maintain reading order
    sorted_list = sorted(cnt_list, key=lambda x: x[1])

    # Combine the extracted text into a single string
    extracted_text = "\n".join([text for _, _, text in sorted_list])

    return extracted_text  # Return the extracted text


def store_album_cover_facesAB(LP_Front_Image_Path: str, LP_Back_Image_Path: str):
    """
    This function stores the text from both the front and back faces of the album cover.
    """
    # Extract text from both images
    front_text = extract_text_from_img(LP_Front_Image_Path)
    back_text = extract_text_from_img(LP_Back_Image_Path)

    # Create a single file to merge both faces' text
    album_name = os.path.basename(LP_Front_Image_Path).split('.')[0]
    merged_file_path = os.path.join(TARGET_DIR, f"{album_name}_combined.txt")

    with open(merged_file_path, "w") as merged_file:
        merged_file.write("Front Cover:\n")
        merged_file.write(front_text)
        merged_file.write("\nBack Cover:\n")
        merged_file.write(back_text)

    print(f"Text from album cover faces stored in {merged_file_path}")


if __name__ == "__main__":
    # Example file paths for testing
    # '/Users/lucazosso/Desktop/Luca_Sandbox_Env/DATA_MEG_PROJ/OCR-AI-Vinyl-Classification/ds/test/20240906_131209.jpg'
    LP_Front_Image_Path = '/Volumes/LaCie_500G/MEG_CAT_TEST/LP2836/LP2836_cover01.jpg'
    # '/Users/lucazosso/Desktop/Luca_Sandbox_Env/DATA_MEG_PROJ/OCR-AI-Vinyl-Classification/ds/test/20240906_131214.jpg'
    LP_Back_Image_Path = '/Volumes/LaCie_500G/MEG_CAT_TEST/LP2836/LP2836_cover02.jpg'

    # Test extract_text_from_img function
    front_text = extract_text_from_img(LP_Front_Image_Path)
    print("Extracted text from front cover:")
    print(front_text)

    back_text = extract_text_from_img(LP_Back_Image_Path)
    print("Extracted text from back cover:")
    print(back_text)

    # Test store_album_cover_facesAB function
    store_album_cover_facesAB(LP_Front_Image_Path, LP_Back_Image_Path)
