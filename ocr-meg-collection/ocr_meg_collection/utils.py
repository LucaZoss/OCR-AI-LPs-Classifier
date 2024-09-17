import logging
import re
from tqdm import tqdm
from functools import wraps
import os
import glob
import sys

from mutagen import File
import numpy as np

# utils.py file for the ocr-meg-collection package

################ Function purely algorithmic ################

# Function for OCR


def fetch_lp_covers_path(base_dir: str, lp_cote_general: str) -> list:
    """
    This function fetches the paths of the front and back covers of an LP from the specified directory.
    It returns a list containing the paths of the front cover and back cover.
    """
    # Fetch file paths in base_dir that match the pattern for Front and Back Covers
    front_cover_paths = glob.glob(
        f"{base_dir}/{lp_cote_general}/*_cover01.jpg")
    back_cover_paths = glob.glob(f"{base_dir}/{lp_cote_general}/*_cover02.jpg")

    # Extract the first path from the glob result or None if no file is found
    front_cover_path = front_cover_paths[0] if front_cover_paths else None
    back_cover_path = back_cover_paths[0] if back_cover_paths else None

    # Create a list with both paths
    cover_paths = [front_cover_path, back_cover_path]

    return cover_paths


##### ORCHESTRATOR CLASS UTILS #####


# utils.py file for the ocr-meg-collection package


def fetch_track_duration(audio_file_path: str) -> str:
    """
    Fetches the track duration from the provided file path.
    """
    try:
        audio = File(audio_file_path)  # Using mutagen to get the audio file
        duration_sec = audio.info.length

        # Format the duration as minutes and seconds with leading zeros for seconds
        minutes = int(duration_sec // 60)
        seconds = int(duration_sec % 60)
        formatted_duration = f"{minutes}'{seconds:02d}"
        return formatted_duration
    except Exception as e:
        logging.error(f"Error fetching duration for {audio_file_path}: {e}")
        return 'missing'


def normalize_track_number(track_number):
    """
    Normalizes track numbers to a common format (e.g., '01').
    Removes non-digit characters and zero-pads to two digits.
    Handles cases where track_number is None or invalid.
    """
    if not isinstance(track_number, str) or not track_number.strip():
        # If track_number is None or not a valid string, log a warning and return None
        logging.warning(f"Invalid track number: {track_number}")
        return None

    # Extract digits from the track number and zero-pad to 2 digits
    match = re.search(r'\d+', str(track_number))
    normalized_number = match.group(0).zfill(2) if match else None

    logging.info(
        f"Normalized track number from '{track_number}' to '{normalized_number}'")
    return normalized_number

# here we should add a logic that if no mapping is found, we should still return the file name and track length


def map_tracks_to_audio_files(track_info, lp_base_dir):
    """
    Maps track names extracted from the AI output to the corresponding audio files.
    If no exact match is found, it falls back to listing available file names and their track lengths.
    """
    audio_map = {}

    # Iterate through all track information
    for track in track_info:
        lp_id = track.get('LP_ID')  # e.g., "LP1234"
        lp_folder = os.path.join(lp_base_dir, lp_id)

        if not os.path.isdir(lp_folder):
            logging.warning(f"Folder {lp_folder} not found for LP_ID: {lp_id}")
            continue

        # Iterate through each file in the LP folder
        for root, _, files in os.walk(lp_folder):
            for file in files:
                if file.endswith('.mp3'):
                    # Extract the face and track number from the file name using regex
                    match = re.match(rf'{lp_id}_([AB])(\d+)\.mp3', file)
                    if match:
                        face = match.group(1)
                        # Normalize track number to two digits
                        track_number = match.group(2).zfill(2)
                        file_path = os.path.join(root, file)
                        audio_map[(lp_id, face, track_number)] = file_path
                        logging.info(
                            f"Matched file: {file_path} for LP_ID: {lp_id}, Face: {face}, Track: {track_number}")

    # Ensure that every track in track_info has a mapped audio file
    for track in track_info:
        lp_id = track.get('LP_ID')
        face = track.get('Face')
        normalized_track_number = track.get('Normalized_Track_Number')

        # Attempt to map each track to its corresponding audio file
        if (lp_id, face, normalized_track_number) in audio_map:
            # If an exact match is found, use it
            file_path = audio_map[(lp_id, face, normalized_track_number)]
            track['File_Name'] = os.path.basename(file_path)
            track['Track_Length'] = fetch_track_duration(file_path)
        else:
            # Fallback: list all files if no exact match is found
            fallback_audio_files = [f for f in os.listdir(
                os.path.join(lp_base_dir, lp_id)) if f.endswith('.mp3')]

            if fallback_audio_files:
                # Use the first available file in the directory as a fallback
                file_path = os.path.join(
                    lp_base_dir, lp_id, fallback_audio_files[0])
                track['File_Name'] = os.path.basename(file_path)
                track['Track_Length'] = fetch_track_duration(file_path)
                logging.info(f"No exact match found for {lp_id}, {face}, {normalized_track_number}. "
                             f"Falling back to {file_path}.")
            else:
                # If no files are found, mark as missing
                track['File_Name'] = 'missing'
                track['Track_Length'] = 'missing'
                logging.warning(f"No audio files found for LP_ID: {lp_id}")

    return audio_map


# UTILITIES FOR TERMINAL DISPLAYING


def with_progress_bar(iterable, desc="Processing", unit="items"):
    """
    A decorator that adds a progress bar to a function processing an iterable.

    Args:
        iterable (iterable): The iterable object to process.
        desc (str): Description for the progress bar.
        unit (str): Unit name to display with each iteration (default is 'items').

    Yields:
        item: Each item from the iterable.
    """
    for item in tqdm(iterable, desc=desc, unit=unit):
        yield item


def with_progress_bar_1(desc="Processing", unit="items"):
    """
    A decorator that adds a progress bar to a function processing an iterable.

    Args:
        desc (str): Description for the progress bar.
        unit (str): Unit name to display with each iteration (default is 'items').

    Returns:
        function: The decorated function with a progress bar.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Call the original function to get the iterable result
            iterable = func(*args, **kwargs)

            # If the result is iterable (list, dict, etc.), show the progress bar
            if hasattr(iterable, '__iter__') and not isinstance(iterable, str):
                return {key: value for key, value in tqdm(iterable.items(), desc=desc, unit=unit)} if isinstance(iterable, dict) else [item for item in tqdm(iterable, desc=desc, unit=unit)]
            else:
                # If the result is not iterable, just return it as is
                return iterable

        return wrapper
    return decorator

##### MAIN.PY FILE UTILS #####


def get_lp_subfolders(base_dir):
    """
    Get a list of subfolder names in the base_dir that start with 'LP'.

    Args:
        base_dir (str): The base directory containing subfolders.

    Returns:
        list: A list of subfolder names starting with 'LP'.
    """
    try:
        subfolders = [name for name in os.listdir(base_dir)
                      if os.path.isdir(os.path.join(base_dir, name)) and name.startswith('LP')]
        return subfolders
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


##### DEPRECATED FUNCTIONS #####
def fetch_track_duration_old(audio_file_path: str) -> str:
    """
    This function fetches the track duration from the provided file path.
    """
    audio = File(audio_file_path)  # using mutagen to get the audio file

    duration_sec = audio.info.length

    # format duration_sec like: 3'01 = 3 minutes and 1 second
    # Format the duration as minutes and seconds with leading zeros for seconds
    minutes = int(duration_sec // 60)
    seconds = int(duration_sec % 60)

    formatted_duration = f"{minutes}'{seconds:02d}"
    return formatted_duration


def list_all_track_duration_from_lp(base_dir: str, lp_cote_general: str) -> dict:
    """
    This function lists all the track durations from a given LP directory and returns a dictionary
    containing the track information for the specified 'cote'. It also fills missing track numbers with NaN.
    """
    directory_path = f"{base_dir}/{lp_cote_general}"  # Updated directory path

    # Get all audio files in the directory
    audio_files = glob.glob(os.path.join(directory_path, "*.mp3"))

    # Extract the 'cote' format (assuming cote format as 'LP2836-1/1')
    cote = f"{lp_cote_general.replace('_', '-').replace('z', '/')}"

    # Initialize the dictionary to hold the information
    track_info = {
        'filename': [],
        'face': [],
        'plage': [],
        'durée': []
    }

    for file in audio_files:
        # Extract the file name
        file_name = os.path.basename(file)

        # Fetch track duration using the custom function
        # Ensure your function returns duration in the format "mm'ss"
        duration = fetch_track_duration(file)

        # Extract side and track number from the file name (format: LP2836_1z1_A01.mp3)
        parts = os.path.splitext(file_name)[0].split("_")
        side_track = parts[-1]
        side = side_track[0]  # 'A' or 'B'
        # Convert '01' to 1, '02' to 2, etc.
        track_number = int(side_track[1:])

        # Append the extracted information to the respective lists
        track_info['filename'].append(file_name)
        track_info['face'].append(side)
        track_info['plage'].append(track_number)
        track_info['durée'].append(duration)

    # Separate sides A and B
    max_tracks = 6
    filled_track_info = {
        'filename': [],
        'face': [],
        'plage': [],
        'durée': []
    }

    # Fill missing tracks for side A
    for i in range(1, max_tracks + 1):
        if i in [track_info['plage'][j] for j in range(len(track_info['plage'])) if track_info['face'][j] == 'A']:
            index = track_info['plage'].index(i)
            filled_track_info['filename'].append(track_info['filename'][index])
            filled_track_info['face'].append('A')
            filled_track_info['plage'].append(i)
            filled_track_info['durée'].append(track_info['durée'][index])
        else:
            filled_track_info['filename'].append(np.nan)
            filled_track_info['face'].append('A')
            filled_track_info['plage'].append(i)
            filled_track_info['durée'].append(np.nan)

    # Fill missing tracks for side B
    for i in range(1, max_tracks + 1):
        if i in [track_info['plage'][j] for j in range(len(track_info['plage'])) if track_info['face'][j] == 'B']:
            index = track_info['plage'].index(i)
            filled_track_info['filename'].append(track_info['filename'][index])
            filled_track_info['face'].append('B')
            filled_track_info['plage'].append(i)
            filled_track_info['durée'].append(track_info['durée'][index])
        else:
            filled_track_info['filename'].append(np.nan)
            filled_track_info['face'].append('B')
            filled_track_info['plage'].append(i)
            filled_track_info['durée'].append(np.nan)

    # Create the final dictionary with 'cote' as the key
    final_dict = {cote: filled_track_info}

    return final_dict
