import cv2
import pytesseract
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

image_path = "/Users/lucazosso/Desktop/Luca_Sandbox_Env/DATA_MEG_PROJ/OCR-AI-Vinyl-Classification/ds/test/20240906_131214.jpg"

# Read the image from which text needs to be extracted
img = cv2.imread(image_path)

# Convert the image to the grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# visualize the grayscale image
plt.imshow(gray, cmap="gray")

# Performing OTSU threshold
ret, thresh1 = cv2.threshold(
    gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# visualize the thresholded image
plt.imshow(thresh1)

# dilation parameter , bigger means less rect
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))

# Applying dilation on the threshold image
dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

# Finding contours
contours, hierarchy = cv2.findContours(
    dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Creating a copy of the image, you can use a binary image as well
im2 = gray.copy()

cnt_list = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # Drawing a rectangle on the copied image
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(im2, (x, y), 8, (255, 255, 0), 8)

    # Cropping the text block for giving input to OCR
    cropped = im2[y:y + h, x:x + w]

    # Apply OCR on the cropped image
    text = pytesseract.image_to_string(cropped)

    cnt_list.append([x, y, text])

    # This list sorts text with respect to their coordinates, in this way texts are in order from top to down
    sorted_list = sorted(cnt_list, key=lambda x: x[1])

    # A text file is created
file = open("recognized2.txt", "w+")
file.write("")
file.close()


for x, y, text in sorted_list:
    # Open the file in append mode
    file = open("recognized2.txt", "a")

    # Appending the text into the file
    file.write(text)
    file.write("\n")

    # Close the file
    file.close()
