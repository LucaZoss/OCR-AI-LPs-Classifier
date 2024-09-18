import pandas as pd
import time
import os


INPUT_DIR = os.path.join(
    os.getcwd(), 'ocr-meg-collection/ds_pipeline/2_raw_csv')
print(INPUT_DIR)

OUTPUT_DIR = os.path.join(
    os.getcwd(), 'ocr-meg-collection/ds_pipeline/3_clean_csv')
print(OUTPUT_DIR)


def get_current_date():
    """Returns the current month, year, and full date in the specified formats."""
    current_month = time.strftime("%m")
    current_year = time.strftime("%Y")
    current_date = time.strftime("%d.%m.%Y")
    return current_month, current_year, current_date


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


def handle_multiple_lps(lp_id, track_info_df):
    """Add a logic to handle multiple LPs in the same folder."""
    # Example: LP123a, LP123b, etc. Extract sub-part logic if applicable
    count_lps = track_info_df[track_info_df['LP_ID'] == lp_id].shape[0]
    return count_lps if count_lps > 1 else 1

# Main functions


def cleaning_labeling_general(general_info_df):
    """Clean and label general info for the central map."""
    current_month, current_year, current_date = get_current_date()
    parrain = 'L. Zosso'

    Lot1_aimp_central_LZO_map = pd.DataFrame({
        'Côte générale': general_info_df['LP_ID'],
        'Localisation': None,
        'Support': 'Disque 33 tours',
        'Format': 'Format 30 cm',
        # Continent based on country
        'Continent': general_info_df['Country'].apply(lambda x: classify_continent(x)[0]),
        # Sub-continent
        'Sub-continent': general_info_df['Country'].apply(lambda x: classify_continent(x)[1]),
        'Pays': general_info_df['Country'].apply(lambda x: x.lower().capitalize() if pd.notnull(x) else x),
        'Région': None,
        'Localité': None,
        'Population': None,
        'Titre': general_info_df['Title'],
        'Sous-Titre': general_info_df['Subtitle'],
        'Traduction': None,
        'Interprète': general_info_df['Performer'],
        'Genre, occasion': None,
        'Instruments': None,
        'Production': general_info_df['Publisher'],
        'Collection': None,
        'Année de production': general_info_df['Publishing Year'],
        'Numéro édition': general_info_df['Label Number'],
        'Auteur du livret': None,
        'Langue': general_info_df['Language'],
        'Photos': None,
        'Pages': None,
        'Collectage': general_info_df['Recording Info'],
        'Commentaire': f'Numérisé en juillet 2024 par Genevay Media Service (Yverdon-les-Bains). ({parrain} {current_month}.{current_year})',
        'date création fiche': current_date,
        'parrain': parrain,
        'Nombre de support': general_info_df.apply(lambda row: handle_multiple_lps(row['LP_ID'], general_info_df), axis=1),
        'Autre pays': NotImplemented,
        'Ancien numéro': None,
        'Correspondance DAT': None,
        'Edition': general_info_df['Label Company'],
        'Année édition': None,
        'Lieu production': None,
        'Lieu édition': None,
        'Numéro de support': None,
        'No Matrice': None,
        # Assuming single LP; modify for logic as needed
        'cote': general_info_df['LP_ID'] + '-1/1',
        'copyrights': 0,
        'exclure de la consultation': 0,
        'ai_info_1_notes': general_info_df['Notes'],
        'ai_info_2_other_info': general_info_df['Other Information']
    })

    return Lot1_aimp_central_LZO_map


def clean_labeling_track(track_info_df):
    """Clean and label track info for the map."""
    Lot1_aimp_plages_LZO_map = pd.DataFrame({
        'Cote': track_info_df['LP_ID'],
        'FileName_TBD': track_info_df['Track Filename'],
        'Face': track_info_df['Face'],
        'Plage': track_info_df['Track Number'],
        'Titre': track_info_df['Track_Name'],
        'Durée': track_info_df['Track Length'],
        'ai_info_track_composer': track_info_df['Track_Composer'],
        'ai_Plage_combined': track_info_df.apply(lambda row: f"{row['Track_Name']}-{row['Track_Composer']}", axis=1)
    })

    return Lot1_aimp_plages_LZO_map


if __name__ == '__main__':

    print("Cleaning and labeling general and track info...")
    general_info_raw_path = os.path.join(INPUT_DIR, 'general_info.csv')
    track_info_raw_path = os.path.join(INPUT_DIR, 'track_info.csv')

    general_info_df = pd.read_csv(general_info_raw_path)
    track_info_df = pd.read_csv(track_info_raw_path)

    general_info_cleaned = cleaning_labeling_general(general_info_df)
    track_info_cleaned = clean_labeling_track(track_info_df)

    general_info_cleaned.to_csv(os.path.join(
        OUTPUT_DIR, 'general_info_cleaned.csv'), index=False)
    track_info_cleaned.to_csv(os.path.join(
        OUTPUT_DIR, 'track_info_cleaned.csv'), index=False)
    print("Cleaning and labeling completed.")
