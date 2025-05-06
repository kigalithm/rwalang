import os
import pandas as pd
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

    # Define default values for metadata columns
    DEFAULT_METADATA_DEFAULTS = {
        "source": "n/a",
        "original_id": "n/a",
        "timestamp": None,
        "is_code_mixed": False,
        "mixed_languages": None,
        "annotator": "n/a",
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


more_kinya = [
    "Bukeye abona agasozi kari gateganye n’urugo rw’Umwami",
    "kakabuza abantu kureba ibituruka kure",
    "Semuhanuka araza abwira Umwami ati.",
    "urampe inka y’ingumba maze nzagukurireho kariya gasozi",
    "nyoroshya mpa ibyo umpa maze uzirorere",
    "Semuhanuka aza yakenyeye bya gasurantambara",
    "yitwaje ikibando, yikoreye ingata nini ifashe ku mutwe wose",
    "araza n’i bwami ahasanga abantu benshi",
    "mbwirira aba bantu bamperekeze hariya hari icyo njya kubabwira",
    "Semuhanuka amaguru aba ariyo abangira ingata asubira ibwami",
    "Ababiri bagiye inama baruta umunani barasana",
    "Ababiri bakika umwe",
    "Ababiri bateranya abeza",
    "Ababiri bica umwe",
    "Ababiri ntibacibwa inka",
    "Abacuranye ubusa basangira ubundi",
    "Abadapfuye ntibabura kubonana",
    "Abagabo babiri ntibabana mu nzu imwe.",
    "Abagabo bararya imbwa zikishyura (zikaryora).",
    "Abagira amenyo baraseka.",
    "Abagira inyonjo bagira ibirori.",
    "Abagira iyo bajya baragenda.",
    "Ab’imbwa bifuza ko budacya.",
    "Aboro babiri ntibasangira umwerera.",
    "Abonye isha itamba ata n’urwo yari yambaye.",
    "Abotanye kera ntibahishanya amabya.",
    "Abo umwami yahaye amata ni bo bamwimye amatwi.",
    "Abwirwa benshi akumva bene yo.",
    "Acuritse inkanda ntacuritse umutima.",
    "Agaca amakungu ni ukwima uwarugendagamo.",
    "Agacumu gahabwa agahari, naho agahararutswe gahabwa agahini.",
    "Agahanga k’umugabo gahangurwa n’uwakaremye.",
    "Agahararo ntikabuza agahararuko.",
    "Agahinda gashira bake.",
    "Agahinda k’inkoko kamenya inkike yatoyemo.",
    "Agahinda k’inkono kamenywa n’uwayiharuye.",
    "Agahinda ntigashira; gashira nyirako yapfuye.",
    "Agahinda ntikajya ahabona.",
    "Agahinda ntikica kagira mubi.",
    "Agahinda ni ukubura uwo ukunda.",
    "Babajije inyamanza bati: ko ufite ukuguru gutoya?",
    "Iti: na ko nagukesheje Rusengo, Semugeshi arenda kukubazamo ubwato!",
    "Babonye umwezi bagira ngo bwakeye",
    "Bagabobarabona yahamagaye urupfu ngo nirugaruke rumurarire",
    "Bagabobarabona yambutse uruzi uruhu rwe rugwamo, ati: ruramaze rwari rurariwe kumeswa",
    "Bagarira yose ntuzi irizera n'irizarumba",
    "Bagaragaza ba Bagaragaza ati; aho abato bari bararyimara",
    "Bakubise nyir'uruhara, nyir'imvi ati: bene ubusembwa batashye",
    "Bakunda inkwi bakanga umushenyi",
    "Bamenya icyo yishe ntibamenya icyo ikijije",
    "Banegura ibyigondoye umuhoro ukarakara",
    "Biza byirabura, bikaza byera, bigaturwa umwami",
    "Bucya zihindura imirishyo",
    "Bucyana ayandi",
    "Gahebeheba ka Ntibaheba, ati: Urwanda rwanyanze ngahata!",
    "Gahima yanywanye na Kajogora, ati: intege nke zitera imico myiza",
    "Gesa ubw'iyo ubw'ino ntiburera",
    "Gihuga isubiye ku gihuru",
    "Gikona burya uzi ko njya ngukeka amababa!",
    "Gira so yiturwa indi.",
    "Guha inda ni ukuyirariza",
    "Guha intanyurwa ni ugusesa",
    "Guherekeza si ukujyana",
    "Gukanura amaso si ko gukira",
    "Gukira kwibagiza gukinga",
    "Gukunda ikitagukunda ni imvura igwa mu ishyamba",
    "Haba ubugabo ntihaba ubukuru.",
    "Haba umugisha ntihaba ubugabo.",
    "Haba umuhanwa hakaba n’umwihane, hakaba n’uwananiranye",
    "Haganya nyir’ubukozwemo naho nyir’ubuteruranwe n’akebo, ntakoma",
    "Hagati y'abapfu n’abapfumu hamenya abahanga",
    "Hagati yumutwe n’umutwaro haca ingata",
    "Haguma kwiha",
    "Haguma umwami ingoma irabazwa",
    "Hakomera imfubyi ifuni iroroha",
    "Agakungu kavamo imbwa yiruka.",
    "Agapfa kabuliwe ni impongo.",
    "Agapfundikiye gatera amatsiko.",
    "Agatinze kazaza ni amenyo ya ruguru.",
    "Ababurana Ari babiri umwe abayigiza nkana",
    "utazi ubwenge ashima ubwe ",
    "ikibuno gishuka amabyi bitari bujyane",
    "Ababiri bishe imbwa y’umwami.",
    "Ababiri ntibacibwa inka.",
    "Ababurana ari babiri umwe aba yigiza nkana.",
    "Abagore bagira inzara ntibagira inzigo.",
    "Abagore baragwira.",
    "Abahigi benshi bayobya imbwa (uburari).",
    "Abakingiranye inyegamo ntibakingirana ingabo.",
    "Abalinzi bajya inama inyoni zijya iyindi.",
    "Abantu bibuka imana iyo amakuba yababanye menshi.",
    "Abasangira bashonje ntawusigariza undi.",
    "Abasangira basigana imbyiro.",
    "Abasangira bike bitana ibisambo.",
    "Abasangira ubusa bitana ibisambo.",
    "Abasobetse imisumbi ntibaba bagihishanye amabya.",
    "Abatanye badatata barasubirana.",
    "Abateranye imigeri ntibahishana amabya.",
    "Aberekeranye ntibabura kwendana.",
    "Abotanye kera ntibahishanya amabya",
    "Abwirwa benshi akumva (akwumvwa na) bene yo.",
    "Agafuni kabagara ubucuti ni akarenge.",
    "Agahwa kari k’uwundi karahandurika.",
    "Agakecuru gahaze gakina n’imyenge y’inzu.",
    "Agakono gashaje karyoshya ibiryo",
    "Agakono gashaje niko karyoshya imboga",
    "Agakungu gakuna imbwa.",
    "Agakungu iyo gashize agashino kayora ivu.",
    "Agakungu kavamo imbwa yiruka.",
    "Agapfa kabuliwe ni impongo.",
    "Agashyize kera gahinyuza inshuti.",
    "Agashyize kera gahinyuza intwari.",
    "Agasozi kagusabye amaraso ntuyakarenza.",
    "Agasozi kamanutse inka kazamuka indi.",
    "Agati gateretswe n’Imana ntigahungabanywa n’umuyaga.",
    "Agatinze kazaryoha ni agatuba k’uruhinja.",
    "Agatinze kazaza ni amenyo ya ruguru.",
    "Ahanze ubwana hamera ubwanwa.",
]
