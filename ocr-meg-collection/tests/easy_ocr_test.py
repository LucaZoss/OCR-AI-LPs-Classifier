import cv2
import easyocr
import os

# Load the image using OpenCV
image_path = "/Volumes/LaCie_500G/MEG_CAT_TEST/LP2836/LP2836_cover01.jpg"

# Check if the file exists
if not os.path.exists(image_path):
    print(f"Error: File does not exist at {image_path}")
    exit()

# Check if the path is a file
if not os.path.isfile(image_path):
    print(f"Error: Path is not a file: {image_path}")
    exit()

# Load the image
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(
        f"Error: Unable to load image at {image_path}. Please check the file path.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Initialize EasyOCR reader for Spanish
reader = easyocr.Reader(['es'])

# Perform text detection on the pre-processed image
results = reader.readtext(thresh, detail=0)

# Check if any text was detected and store the first detected text
if results:
    first_detected_text = results[0]
    print(f"First detected text: {first_detected_text}")
else:
    print("No text detected.")
