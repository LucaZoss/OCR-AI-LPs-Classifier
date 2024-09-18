import pandas as pd
import time
import os


class Cleaner:
    def __init__(self, input_dir, output_dir):
        """
        Initializes the Cleaner class with input and output directories.

        Args:
            input_dir (str): Path to the input directory containing raw CSV files.
            output_dir (str): Path to the output directory for cleaned CSV files.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir

        # Define paths for general info and track info CSVs
        self.general_info_raw_path = os.path.join(
            self.input_dir, 'general_info.csv')
        self.track_info_raw_path = os.path.join(
            self.input_dir, 'track_info.csv')

        # Read the CSV files
        self.general_info_df = pd.read_csv(self.general_info_raw_path)
        self.track_info_df = pd.read_csv(self.track_info_raw_path)

    @staticmethod
    def get_current_date():
        """Returns the current month, year, and full date in the specified formats."""
        current_month = time.strftime("%m")
        current_year = time.strftime("%Y")
        current_date = time.strftime("%d.%m.%Y")
        return current_month, current_year, current_date

    @staticmethod
    def classify_continent(country):
        """Classify the country into a continent and sub-continent."""
        if country in ['Spain', 'Espagne']:
            return 'Europe', 'Europe méridionale'
        elif country in ['Uruguay', 'URUGUAY', 'Paraguay', 'Bolivia', 'Bolivie']:
            return 'Amérique', 'Amérique du Sud'
        elif country in ['Mexico', 'Mexique']:
            return 'Amérique', 'Amérique centrale'
        else:
            return None, None

    def handle_multiple_lps(self, lp_id):
        """Add a logic to handle multiple LPs in the same folder."""
        count_lps = self.track_info_df[self.track_info_df['LP_ID']
                                       == lp_id].shape[0]
        return count_lps if count_lps > 1 else 1

    def cleaning_labeling_general(self):
        """Clean and label general info for the central map."""
        current_month, current_year, current_date = self.get_current_date()
        parrain = 'L. Zosso'

        Lot1_aimp_central_LZO_map = pd.DataFrame({
            'Côte générale': self.general_info_df['LP_ID'],
            'Localisation': None,
            'Support': 'Disque 33 tours',
            'Format': 'Format 30 cm',
            'Continent': self.general_info_df['Country'].apply(lambda x: self.classify_continent(x)[0]),
            'Sub-continent': self.general_info_df['Country'].apply(lambda x: self.classify_continent(x)[1]),
            'Pays': self.general_info_df['Country'].apply(lambda x: x.lower().capitalize() if pd.notnull(x) else x),
            'Région': None,
            'Localité': None,
            'Population': None,
            'Titre': self.general_info_df['Title'],
            'Sous-Titre': self.general_info_df['Subtitle'],
            'Traduction': None,
            'Interprète': self.general_info_df['Performer'],
            'Genre, occasion': None,
            'Instruments': None,
            'Production': self.general_info_df['Publisher'],
            'Collection': None,
            'Année de production': self.general_info_df['Publishing Year'],
            'Numéro édition': self.general_info_df['Label Number'],
            'Auteur du livret': None,
            'Langue': self.general_info_df['Language'],
            'Photos': None,
            'Pages': None,
            'Collectage': self.general_info_df['Recording Info'],
            'Commentaire': f'Numérisé en juillet 2024 par Genevay Media Service (Yverdon-les-Bains). ({parrain} {current_month}.{current_year})',
            'date création fiche': current_date,
            'parrain': parrain,
            'Nombre de support': self.general_info_df.apply(lambda row: self.handle_multiple_lps(row['LP_ID']), axis=1),
            'Autre pays': NotImplemented,
            'Ancien numéro': None,
            'Correspondance DAT': None,
            'Edition': self.general_info_df['Label Company'],
            'Année édition': None,
            'Lieu production': None,
            'Lieu édition': None,
            'Numéro de support': None,
            'No Matrice': None,
            'cote': self.general_info_df['LP_ID'] + '-1/1',
            'copyrights': 0,
            'exclure de la consultation': 0,
            'ai_info_1_notes': self.general_info_df['Notes'],
            'ai_info_2_other_info': self.general_info_df['Other Information']
        })

        return Lot1_aimp_central_LZO_map

    def clean_labeling_track(self):
        """Clean and label track info for the map."""
        Lot1_aimp_plages_LZO_map = pd.DataFrame({
            'Cote': self.track_info_df['LP_ID'],
            'FileName_TBD': self.track_info_df['Track Filename'],
            'Face': self.track_info_df['Face'],
            'Plage': self.track_info_df['Track Number'],
            'Titre': self.track_info_df['Track_Name'],
            'Durée': self.track_info_df['Track Length'],
            'ai_info_track_composer': self.track_info_df['Track_Composer'],
            'ai_Plage_combined': self.track_info_df.apply(lambda row: f"{row['Track_Name']}-{row['Track_Composer']}", axis=1)
        })

        return Lot1_aimp_plages_LZO_map

    def run_cleaner(self):
        """Runs the cleaner process and saves cleaned CSVs."""
        general_info_cleaned = self.cleaning_labeling_general()
        track_info_cleaned = self.clean_labeling_track()

        # Save the cleaned general info and track info CSVs
        general_info_cleaned.to_csv(os.path.join(
            self.output_dir, 'general_info_cleaned.csv'), index=False)
        track_info_cleaned.to_csv(os.path.join(
            self.output_dir, 'track_info_cleaned.csv'), index=False)
        print("Cleaning and labeling completed.")


if __name__ == '__main__':
    INPUT_DIR = os.path.join(
        os.getcwd(), 'ocr-meg-collection', 'ds_pipeline', '2_raw_csv')
    OUTPUT_DIR = os.path.join(
        os.getcwd(), 'ocr-meg-collection', 'ds_pipeline', '3_clean_csv')

    print("Input Directory:", INPUT_DIR)
    print("Output Directory:", OUTPUT_DIR)

    # Initialize and run the cleaner
    cleaner_instance = Cleaner(INPUT_DIR, OUTPUT_DIR)
    cleaner_instance.run_cleaner()
