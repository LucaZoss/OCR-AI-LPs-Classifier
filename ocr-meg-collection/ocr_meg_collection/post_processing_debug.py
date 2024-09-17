import os
import json
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from ocr_meg_collection.utils import fetch_track_duration, map_tracks_to_audio_files, normalize_track_number
import re

# Setup I/O Directories
INPUT_DIR = os.path.join(
    os.getcwd(), 'ocr-meg-collection', 'ds_pipeline', '1_json_inf_outputs')
TARGET_DIR = os.path.join(
    os.getcwd(), 'ocr-meg-collection', 'ds_pipeline', '2_raw_csv')


class Orchestrator:
    def __init__(self, input_dir=INPUT_DIR, output_dir=TARGET_DIR, lp_base_dir=None, max_workers=4):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.lp_base_dir = lp_base_dir
        self.max_workers = max_workers

    def split_info(self):
        # Split JSON files into General Info and Track Info
        general_info = []
        track_info = []

        # Iterate through 1_json_inf_outputs directory to get each LP AI outputs
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.json'):
                    json_file_path = os.path.join(root, file)
                    logging.info(f"Processing JSON file: {json_file_path}")
                    with open(json_file_path, 'r') as f:
                        try:
                            data = json.load(f)

                            # Extract General Information
                            general_info_entry = data.get(
                                'General Information', {})
                            if general_info_entry:
                                # Add LP_ID fetched from the filename
                                general_info_entry['LP_ID'] = self.fetch_lp_id(
                                    json_file_path)
                                # Ensure "country" field is included
                                general_info_entry['country'] = general_info_entry.get(
                                    'country', 'missing')
                                general_info.append(general_info_entry)

                            # Extract Track Information
                            track_info_entries = data.get('Track Info', [])
                            if track_info_entries:
                                for track in track_info_entries:
                                    # Add LP_ID fetched from the filename
                                    track['LP_ID'] = self.fetch_lp_id(
                                        json_file_path)
                                    # Ensure all fields are filled
                                    track.setdefault('Track_Name', 'missing')
                                    track.setdefault(
                                        'Track_Composer', 'missing')
                                    track.setdefault('Track_Length', 'missing')
                                    track_info.append(track)

                        except json.JSONDecodeError:
                            logging.error(
                                f"Error decoding JSON file: {json_file_path}")
        return general_info, track_info

    def fetch_lp_id(self, json_file_path):
        """
        Fetches the LP_ID from the JSON file path.
        """
        return os.path.basename(json_file_path).split('_')[0]

    def merge_info(self, general_info, track_info):
        # Create a DataFrame for General Information
        general_info_df = pd.DataFrame(general_info)

        # Create a DataFrame for Track Information
        track_info_df = pd.DataFrame(track_info)

        # Ensure LP_ID is included in both dataframes
        if 'LP_ID' not in general_info_df.columns:
            general_info_df['LP_ID'] = general_info_df.apply(
                lambda row: self.fetch_lp_id(row.name), axis=1
            )
        if 'LP_ID' not in track_info_df.columns:
            track_info_df['LP_ID'] = track_info_df.apply(
                lambda row: self.fetch_lp_id(row.name), axis=1
            )

        # Normalize column names to match the AI output
        if 'Track Number' not in track_info_df.columns and 'Track_Number' in track_info_df.columns:
            track_info_df.rename(
                columns={'Track_Number': 'Track Number'}, inplace=True)

        # Use ThreadPoolExecutor to parallelize the fetching of filenames and track lengths
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_track_info = {
                executor.submit(self.get_track_details, row): row for _, row in track_info_df.iterrows()
            }

            # Update track info with results from the threads
            for future in as_completed(future_to_track_info):
                row_index = future_to_track_info[future].name
                try:
                    track_filename, track_length = future.result()
                    track_info_df.at[row_index,
                                     'Track Filename'] = track_filename
                    track_info_df.at[row_index, 'Track Length'] = track_length
                except Exception as e:
                    logging.error(f"Error processing track info: {e}")

        # Save the data to CSV files
        general_info_csv_path = os.path.join(
            self.output_dir, 'general_info.csv')
        track_info_csv_path = os.path.join(self.output_dir, 'track_info.csv')

        general_info_df.to_csv(general_info_csv_path, index=False)
        track_info_df.to_csv(track_info_csv_path, index=False)

        logging.info(f"General Information saved to: {general_info_csv_path}")
        logging.info(f"Track Information saved to: {track_info_csv_path}")

        return general_info_csv_path, track_info_csv_path

    def get_track_details(self, row):
        """
        Fetches the filename and track length for each row.
        """
        lp_id = row['LP_ID']
        face = row.get('Face', '')
        track_number = normalize_track_number(row.get('Track Number', ''))

        # Construct the expected filename based on the LP_ID, Face, and Track Number
        constructed_filename = f"{lp_id}_1z1_{face}{track_number}.mp3"
        constructed_filepath = os.path.join(
            self.lp_base_dir, lp_id, constructed_filename)

        if os.path.exists(constructed_filepath):
            # If the file exists, use it
            track_filename = constructed_filename
            track_length = fetch_track_duration(constructed_filepath)
        else:
            # If no exact match is found, fallback to the first available file
            track_filename = constructed_filename
            track_length = 'missing'

        return track_filename, track_length

    def run(self):
        # Main execution function
        general_info, track_info = self.split_info()

        # Merge the extracted information and save to CSV
        self.merge_info(general_info, track_info)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Define the base directory where LP subfolders reside
    LP_BASE_DIR = '/Volumes/T7'  # Update as needed

    orchestrator = Orchestrator(lp_base_dir=LP_BASE_DIR)
    orchestrator.run()
