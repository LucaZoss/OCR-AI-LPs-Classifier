import os
import json
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from ocr_meg_collection.utils import fetch_track_duration, map_tracks_to_audio_files, normalize_track_number

# Setup I/O Directories
INPUT_DIR = os.path.join(
    os.getcwd(), 'ocr-meg-collection', 'ds_pipeline', '1_json_inf_outputs')
TARGET_DIR = os.path.join(
    os.getcwd(), 'ocr-meg-collection', 'ds_pipeline', '2_raw_csv')


class Orchestrator:
    def __init__(self, input_dir=INPUT_DIR, output_dir=TARGET_DIR, lp_base_dir=None, max_workers=4):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.lp_base_dir = lp_base_dir  # Base directory where LP subfolders reside
        self.max_workers = max_workers

    def split_info(self):
        # Split JSON files into General Info and Track Info
        general_info = []
        track_info = []

        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.json'):
                    with open(os.path.join(root, file), 'r') as f:
                        try:
                            data = json.load(f)

                            # Extract General Information
                            general_info_entry = data.get(
                                'General Information', {})
                            general_info.append(general_info_entry)

                            # Extract LP_ID from General Information
                            lp_id = general_info_entry.get(
                                'LP_ID', '').replace(' ', '')

                            # Extract Track Info and assign LP_ID to each track
                            tracks = data.get('Track Info', [])
                            for track in tracks:
                                # Assign LP_ID to each track
                                track['LP_ID'] = lp_id
                                track_info.append(track)

                        except json.JSONDecodeError as e:
                            logging.error(
                                f"Error decoding JSON file {file}: {e}")
                        except Exception as e:
                            logging.error(
                                f"Unexpected error while processing file {file}: {e}")

        return general_info, track_info

    def merge_json_to_csv(self, json_list, output_file):
        # Merge JSON files into a single CSV using pandas
        if not json_list:
            logging.warning(f"No data found for merging into {output_file}")
            return

        df = pd.DataFrame(json_list)
        # Add the new column 'Cat_Generated_By_AI' with value True
        df['AI_Cat_Generated'] = True
        df.to_csv(output_file, index=False, encoding='utf-8')
        logging.info(f"Data merged into {output_file}")

    def map_tracks(self, track_info):
        # Normalize LP_IDs to match folder names
        for track in track_info:
            track['LP_ID'] = track.get('LP_ID', '').replace(' ', '')

            # Check if LP_ID is missing
            if not track['LP_ID']:
                logging.error(f"Missing LP_ID in track info: {track}")
                continue

            # Normalize track numbers to a common format
            track['Normalized_Track_Number'] = normalize_track_number(
                track.get('Track_Number', ''))

            if not track['Normalized_Track_Number']:
                logging.error(
                    f"Error normalizing track number for track: {track}")
                continue

            logging.info(
                f"Processing track: LP_ID={track['LP_ID']}, Face={track['Face']}, Normalized Track Number={track['Normalized_Track_Number']}")

        # Map track names to corresponding audio files
        track_map = map_tracks_to_audio_files(track_info, self.lp_base_dir)

        updated_tracks = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_track = {
                executor.submit(self.update_track_info, track, track_map): track for track in track_info
            }

            for future in as_completed(future_to_track):
                try:
                    updated_track = future.result()
                    updated_tracks.append(updated_track)
                except Exception as e:
                    logging.error(f"Error processing track: {e}")

        return updated_tracks

    def update_track_info(self, track, track_map):
        # Use normalized track number for lookup
        audio_file_path = track_map.get(
            (track['LP_ID'], track['Face'], track['Normalized_Track_Number']))
        file_name = os.path.basename(
            audio_file_path) if audio_file_path else 'missing'
        track_length = fetch_track_duration(
            audio_file_path) if audio_file_path else 'missing'

        return {
            'File_Name': file_name,
            'Face': track['Face'],
            'Track_Number': track['Track_Number'],
            'Track_Name': track['Track_Name'],
            'Track_Composer': track['Track_Composer'],
            'Track_Length': track_length,
            'Track_Length_AI': track.get('Track_Length', 'missing')
        }

    def run(self):
        # Main execution function
        general_info, track_info = self.split_info()

        # Merge General Info to CSV
        general_info_csv = os.path.join(self.output_dir, 'general_info.csv')
        self.merge_json_to_csv(general_info, general_info_csv)

        # Merge Track Info to CSV
        track_info = self.map_tracks(track_info)
        track_info_csv = os.path.join(self.output_dir, 'track_info.csv')
        self.merge_json_to_csv(track_info, track_info_csv)

        logging.info("Orchestration completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Define the base directory where LP subfolders reside
    LP_BASE_DIR = '/Volumes/LaCie_500G/MEG_CAT_TEST'  # Example path, update as needed

    orchestrator = Orchestrator(lp_base_dir=LP_BASE_DIR)
    orchestrator.run()
