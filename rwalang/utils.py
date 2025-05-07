import os
import pandas as pd
import datetime
from config import logging, DEFAULT_OUTPUT_CSV_PATH


def save_training_data_to_csv(
    training_data_dict, output_csv_path=DEFAULT_OUTPUT_CSV_PATH
):
    """
    Converts training data from dictionary format {label: [texts]} to a pandas DataFrame
    and saves it to a CSV file with 'text', 'language', and other metadata columns
    filled with default values.

    Args:
        training_data_dict (dict): A dictionary where keys are language labels (str)
                                   and values are lists of text samples (list of str).
        output_csv_path (str): Path where the output CSV file will be saved.
                               The directory structure for the output path must exist.

    Returns:
        pandas.DataFrame: The DataFrame created (also saved to CSV).
                          Returns None if the input dictionary is empty or contains no samples.
    """
    if not training_data_dict:
        print(
            "Warning: Input training data dictionary is empty."
        )  # Use print for simplicity
        return None

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Define default values for metadata columns
    DEFAULT_METADATA_DEFAULTS = {
        "source": "n/a",
        "original_id": "n/a",
        "timestamp": timestamp,
        "is_code_mixed": False,
        "mixed_languages": None,
        "annotator": "kigalithm",
        "quality_score": None,
    }

    data_for_df = []

    # Loop through each language (label) and its list of texts
    for lang, texts_list in training_data_dict.items():
        # Loop through each individual text sample in the list
        for text in texts_list:
            # Create a dictionary for this specific row/sample
            row_dict = {}

            # Add the essential columns
            row_dict["text"] = text
            row_dict["language"] = lang

            # Add all default metadata columns
            row_dict.update(DEFAULT_METADATA_DEFAULTS)

            # Append the complete row dictionary to our list
            data_for_df.append(row_dict)

    if not data_for_df:
        print("Warning: No text samples found in the input dictionary lists.")
        return None

    # Create pandas DataFrame from the list of dictionaries
    df = pd.DataFrame(data_for_df)

    # Optional: Define and reorder columns for clarity
    # Ensure 'text' and 'language' are first, followed by the rest
    all_expected_columns = ["text", "language"] + list(DEFAULT_METADATA_DEFAULTS.keys())
    df = df[all_expected_columns]  # This also handles potential order

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Save DataFrame to CSV
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    logging.info(
        f"Successfully saved data to CSV at {output_csv_path} with {len(df)} samples."
    )

    return df


if __name__ == "__main__":

    eac_df = pd.read_csv("../data/raw/eac.csv", encoding="utf-8")
    training_df = pd.read_csv("../data/raw/training_data.csv", encoding="utf-8")
    COMBINED_CSV_PATH = "../data/raw/training_data_spoken_langs_in_rw.csv"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    EAC_COLUMN_MAPPING = {"Text": "text", "Language": "language"}

    # Define default values for columns that might be missing in the EAC data
    DEFAULT_METADATA_DEFAULTS = {
        "source": "eac_dataset",  # Default source for EAC data
        "original_id": "n/a_eac",  # Default ID prefix for EAC data
        "timestamp": timestamp,
        "is_code_mixed": False,
        "mixed_languages": None,
        "annotator": "kigalithm",  # Default annotator for EAC data
        "quality_score": None,
    }

    # --- Data Loading ---

    rw_eac_df = eac_df[eac_df["Language"] == "Kinyarwanda"]
    sw_eac_df = eac_df[eac_df["Language"] == "Kiswahili"]
    fr_eac_df = eac_df[eac_df["Language"] == "French"]
    en_eac_df = eac_df[eac_df["Language"] == "English"]

    TARGET_COLUMNS = [
        "text",
        "language",
        "source",
        "original_id",
        "timestamp",
        "is_code_mixed",
        "mixed_languages",
        "annotator",
        "quality_score",
    ]

    sw_eac_df = sw_eac_df.rename(columns=EAC_COLUMN_MAPPING, errors='ignore')
    rw_eac_df = rw_eac_df.rename(columns=EAC_COLUMN_MAPPING, errors='ignore')
    en_eac_df = en_eac_df.rename(columns=EAC_COLUMN_MAPPING, errors='ignore')
    fr_eac_df = fr_eac_df.rename(columns=EAC_COLUMN_MAPPING, errors='ignore')

    # Swahili
    if 'language' in sw_eac_df.columns:
     sw_eac_df['language'] = 'swahili'
     sw_eac_df['timestamp'] = timestamp
     sw_eac_df['annotator'] = 'kigalithm'
    else:
     print("Error: 'language' column not found in Swahili data after renaming.")


    # Kinyarwanda
    if 'language' in rw_eac_df.columns:
     rw_eac_df['language'] = 'kinyarwanda'
     rw_eac_df['timestamp'] = timestamp
     rw_eac_df['annotator'] = 'kigalithm'
    else:
     print("Error: 'language' column not found in Kinyarwanda data after renaming.")


    # English
    if 'language' in en_eac_df.columns:
     en_eac_df['language'] = 'english'
     en_eac_df['timestamp'] = timestamp
     en_eac_df['annotator'] = 'kigalithm'
    else:
     print("Error: 'language' column not found in English data after renaming.")


    # French
    if 'language' in fr_eac_df.columns:
     fr_eac_df['language'] = 'french'
     fr_eac_df['timestamp'] = timestamp
     fr_eac_df['annotator'] = 'kigalithm'
    else:
     print("Error: 'language' column not found in French data after renaming.")


    # Swahili
    for col in TARGET_COLUMNS:
        if col not in sw_eac_df.columns:
            default_value = DEFAULT_METADATA_DEFAULTS.get(col, None)
            sw_eac_df[col] = default_value


    # Kinyarwanda
    for col in TARGET_COLUMNS:
        if col not in rw_eac_df.columns:
            default_value = DEFAULT_METADATA_DEFAULTS.get(col, None)
            rw_eac_df[col] = default_value


    # English
    for col in TARGET_COLUMNS:
        if col not in en_eac_df.columns:
            default_value = DEFAULT_METADATA_DEFAULTS.get(col, None)
            en_eac_df[col] = default_value


    # French
    for col in TARGET_COLUMNS:
        if col not in fr_eac_df.columns:
            default_value = DEFAULT_METADATA_DEFAULTS.get(col, None)
            fr_eac_df[col] = default_value



    # sw_eac_df = sw_eac_df[TARGET_COLUMNS]

    combined_training_df = pd.concat([training_df, sw_eac_df, rw_eac_df, en_eac_df, fr_eac_df], ignore_index=True)
    
    sw_df = combined_training_df[combined_training_df['language'] == 'swahili']
    rw_df = combined_training_df[combined_training_df['language'] == 'kinyarwanda']
    fr_df = combined_training_df[combined_training_df['language'] == 'french']
    en_df = combined_training_df[combined_training_df['language'] == 'english']
    
    print(f"sw: {len(sw_df)} vs rw:{len(rw_df)} vs en: {len(en_df)} vs fr: {len(fr_df)}")
    combined_training_df.to_csv("../data/raw/spoken_languages_in_rw.csv", index=False, encoding='utf-8')

