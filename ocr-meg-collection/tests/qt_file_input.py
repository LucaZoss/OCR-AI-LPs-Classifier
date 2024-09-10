import os
import glob

# Define the directory path where your files are located
directory_path = "/Users/lucazosso/Desktop/Luca_Pro/Publicis"
directory_path_1 = "/Users/lucazosso/Desktop/Luca_Pro/Publicis/SEA"

# r"C:\Users\YourUsername\Documents\DataFolder"

# List all files in the directory (e.g., all CSV files)
# Adjust the pattern for different file types
file_pattern = os.path.join(
    directory_path, "*.xlsx") | os.path.join(directory_path_1, "*.xls")
file_list = glob.glob(file_pattern)

# Print the list of files
print("Files found:", file_list)
