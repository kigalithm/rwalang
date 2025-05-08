# kgt-rwalang

An enhanced language detector for Kinyarwanda, designed to accurately identify Kinyarwanda text, including code-mixed content involving English, French, and Swahili.

> **Note:** It is not perfect yet, but it is the best there is, and it works.

## Overview

kgt-rwalang combines traditional methods like character n-grams and TF-IDF with Kinyarwanda-specific linguistic features to achieve robust language detection. It includes specialized handling and confidence scoring for text that mixes Kinyarwanda with common foreign loan words and grammatical structures.

## Features

* **Character N-grams & TF-IDF:** Standard text feature extraction.
* **Kinyarwanda Linguistic Features:** Incorporates analysis of ibihekane, ibyungo, accented vowels, grammatical markers, and common affixes.
* **Code-Mixing Detection:** Includes logic to identify and better handle text containing a mix of Kinyarwanda and foreign language elements (loan words, grammatical patterns).
* **Ensemble Model:** Uses a combination of machine learning classifiers for improved accuracy.
* **Model Persistence:** Ability to save and load trained models using `joblib`.
* **Configurable Thresholds:** Adjust detection sensitivity.

## Installation

You can install the package using pip:

```bash
pip install kgt-rwalang
```

## Usage

Here's a basic example of how to use the `KinyaLangDetector` class:

```python
import pandas as pd
from rwalang.detector import KinyaLangDetector

# Instantiate the Detector
detector = KinyaLangDetector()

# Load or Train the Model ---

try:
    # Attempt to load the model.
    # Call load_model as is or pass a path to your own model.
    detector.load_model()
except FileNotFoundError:
    print(f"Model file not found. Training new model...")
    try:
        training_df = pd.read_csv('path/to/your_training_data.csv', encoding='utf-8')
    except FileNotFoundError:
         print("Error: Training data CSV not found. Cannot train model.")
         # Handle this error - maybe exit or get data another way
         training_df = None

    if training_df is not None:
        # Train the model using your DataFrame
        detector.train(training_df)
        print("Training complete.")

        # Save the newly trained model for future use
        # User might save their trained model to a specific path they control
        detector.save_model(model_filepath)
        print(f"Model saved to {model_filepath}")
except Exception as e:
    # Catch any other unexpected errors during loading
    print(f"An unexpected error occurred during model handling: {e}")
    # The detector.model might be None here

# Use the Detector ---

if detector.model: # Check if a model is successfully loaded or trained
    test_texts = [
        "Muraho, amakuru yanyu?",      # Kinyarwanda
        "Ese ubu yamaze gu testing?",  # Code-mixed Kinyarwanda/English
        "Hello, how are you?",         # Pure English
        "Je parle français",           # Pure French
        "Habari zenu?",                # Pure Swahili
        "Mfite message kuri whatsapp.", # Code-mixed Kinyarwanda/English
        "Ibyo ni sawa kabisa.",        # Code-mixed Kinyarwanda/Swahili
        "Ni mwicare se banyabusa!",    # Pure Kinyarwanda
        "Erega ntabwo mbyigoragoraho rwose, Ninjye wishe umuzungu!"     # Pure Kinyarwanda
    ]

    for text in test_texts:
        result = detector.detect(text)
        print(f"Input: '{text}'")
        print(f"Detection: {result}")
        # Result will be a dictionary like:
        # {
        #   'language': 'kinyarwanda' or 'english' etc.,
        #   'confidence': 0.0 to 1.0,
        #   'is_kinyarwanda': True or False,
        #   'code_mixed': True or False,
        #   'mix_info': {... details if code_mixed},
        #   'analysis_scores': {... additional scores}
        # }
        print("-" * 20)
else:
    print("Detector model is not available. Please train or load a model first.")
```

## Training Data

To train the model, you will need a dataset of text samples labelled with their language. The ```detector.train()``` method is designed to accept a dictionary with at least a 'text' key (containing the text samples) and a 'language' key (containing the corresponding language labels).

A recommended format for your training data CSV is:
```
text,language,source,original_id,timestamp,is_code_mixed,mixed_languages,annotator,quality_score
"Muraho, amakuru?",kinyarwanda,manual,k_001,2024-01-01,False,None,UserA,5
"Ese ubu urakora project?",kinyarwanda,social,tw_123,2024-03-15,True,english,AutoCollect,4
"Hello, how are you?",english,manual,e_001,2024-01-01,False,None,UserA,5
...
```

## Licence

This project is licensed under the [MIT License](https://opensource.org/license/mit). See the ```LICENSE``` file for details.


## Contributing

We welcome contributions to kgt_rwalang! If you have suggestions for improvements, bug fixes, or want to add more linguistic features or data, please follow these steps:

1. Fork the repository on GitHub
2. Create a new branch for your feature or bugfix.
3. Make your changes, ensuring your code follows the project's style and includes appropriate tests (if applicable).
4. Write clear commit messages.
5. Push your branch to your fork.
6. Submit a pull request to the main repository, describing your changes.

Please feel free to open an issue first to discuss larger changes.


## Author

[izamha](https://github.com/izamha)

```giliza@kigalithm.com```


## Sponsor

[Kigalithm](https://kigalithm.com)

```foss@kigalithm.com```

```intern@kigalithm.com```
