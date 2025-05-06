import re
import random
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from .linguistic_features import KinyaLinguisticFeatures


warnings.filterwarnings('ignore')


class EnhancedKinyaLangDetector:
    """
    An enhanced language detector for Kinyarwanda that combines character n-grams,
    TF-IDF vectorization, and Kinyarwanda-specific linguistic features to identify
    if text is in Kinyarwanda, with improved handling of code-mixed text.
    """

    def __init__(self):
        self.model = None
        self.ngram_range = (1, 6)  # Use character n-grams from 1 to 6
        # Use a consistent list of languages, include 'other'
        self.languages = ['kinyarwanda', 'english', 'french', 'swahili', 'other']
        self.feature_extractor = None
        # Instantiate the refactored KinyarwandaLinguisticFeatures
        self.linguistic_features = KinyaLinguisticFeatures()

        # NEW: Threshold for code-mixed text classification
        self.mixed_text_threshold = 0.65


    def preprocess_text(self, text):
        """Basic text preprocessing to standardize input"""
        if not text:
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def train(self, training_data):
        """
        Train the enhanced language detection model combining traditional n-gram approach
        with Kinyarwanda-specific linguistic features

        Parameters:
        training_data: dict with keys for each language containing a list of text samples
        """
        # Prepare data
        X = []
        y = []

        # Filter training_data to only include languages we intend to model
        filtered_training_data = {lang: texts for lang, texts in training_data.items() if lang in self.languages}

        for lang, texts in filtered_training_data.items():
            for text in texts:
                X.append(self.preprocess_text(text))
                y.append(lang)

        # Augment Kinyarwanda data by creating synthetic samples (optional but helpful)
        # Note: The augmentation in the original code is simple character mixing.
        # A more robust augmentation might use the linguistic features class's ability
        # to identify loan words or grammar patterns to create more realistic mixed text.
        # Let's keep the original augmentation logic for now unless we refactor it.
        # The original augmentation was in _augment_data, which we will keep.
        X_augmented, y_augmented = self._augment_data(X, y)


        # Split into training and validation sets
        # Ensure stratification handles cases where a language might have very few samples
        # or is absent after filtering.
        valid_indices = [i for i, label in enumerate(y_augmented) if label in self.languages]
        X_filtered = [X_augmented[i] for i in valid_indices]
        y_filtered = [y_augmented[i] for i in valid_indices]

        if len(X_filtered) < 2 or len(set(y_filtered)) < 2:
             print("Not enough data or languages after filtering for train/test split.")
             # Potentially train on all data if validation split isn't feasible
             X_train, y_train = X_filtered, y_filtered
             X_val, y_val = [], [] # No separate validation set
        else:
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
                )
            except ValueError as e:
                 print(f"Could not stratify split. Falling back to simple split: {e}")
                 X_train, X_val, y_train, y_val = train_test_split(
                    X_filtered, y_filtered, test_size=0.2, random_state=42
                )


        # Create our base feature extractor (TF-IDF)
        self.feature_extractor = TfidfVectorizer(
            analyzer='char',
            ngram_range=self.ngram_range,
            max_features=20000, # Consider adjusting based on dataset size
            sublinear_tf=True,
            min_df=2,
            use_idf=True
        )

        # Create base classifiers for the ensemble
        # Ensure class_weight is used for imbalanced datasets, especially with augmentation
        nb_classifier = MultinomialNB(alpha=0.01) # MNBN doesn't support class_weight, but alpha helps
        svm_classifier = SVC(C=10, kernel='linear', probability=True, class_weight='balanced')
        rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            class_weight='balanced',
            random_state=42 # for reproducibility
        )

        # Create the enhanced pipeline with linguistic features
        # This FeatureUnion takes the raw text and passes it to both
        # the TF-IDF vectorizer and the KinyarwandaLinguisticFeatures transformer.
        combined_features = FeatureUnion([
            ('tfidf', self.feature_extractor),
            ('linguistic', self.linguistic_features)
        ])

        # Create and train the ensemble model with combined features
        self.model = Pipeline([
            ('features', combined_features),
            ('classifier', VotingClassifier(
                estimators=[
                    ('nb', nb_classifier),
                    ('svm', svm_classifier),
                    ('rf', rf_classifier)
                ],
                voting='soft', # Use soft voting for probability combination
                weights=[1, 1, 1] # Adjust weights based on classifier performance if needed
            ))
        ])

        # Perform grid search to find optimal parameters
        # NOTE: Grid search on a pipeline with FeatureUnion and VotingClassifier
        # can be very computationally expensive, especially with linguistic features
        # which might add overhead per sample. Use a small grid or skip on large data.
        param_grid = {
            # Example parameters for TF-IDF
            # 'features__tfidf__max_features': [10000, 20000], # Keep it reasonable
            # 'features__tfidf__ngram_range': [(1, 5), (1, 6)],
            # Example parameters for Naive Bayes (within VotingClassifier)
            'classifier__nb__alpha': [0.01, 0.05], # Tune alpha for Naive Bayes

            # Example parameters for SVM (within VotingClassifier)
            # 'classifier__svm__C': [1, 10], # Tune C for SVM

            # Example parameters for RF (within VotingClassifier)
            # 'classifier__rf__n_estimators': [50, 100], # Tune n_estimators for RF

            # Example parameters for VotingClassifier weights (tune if necessary)
            # 'classifier__weights': [[1, 1, 1], [1, 2, 1], [2, 1, 1]] # weights for (nb, svm, rf)
        }

        # Simplify grid search significantly if dataset is large
        if len(X_train) > 500: # Adjusted threshold
             print("Large dataset detected. Using simplified parameter optimization.")
             param_grid = {
                 'classifier__nb__alpha': [0.01, 0.05], # Focus on NB as it's common in text
             }
        if len(X_train) > 2000:
             print("Very large dataset detected. Skipping grid search.")
             # Just train the model with default or chosen parameters
             self.model.fit(X_train, y_train)
             print("Model trained with default parameters.")
        else:
            # Perform grid search
            print("Performing grid search for optimal parameters...")
            grid_search = GridSearchCV(
                self.model,
                param_grid,
                cv=3, # Reduced CV folds for speed
                scoring='accuracy',
                verbose=1,
                n_jobs=-1 # Use all available cores
            )

            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_

            print(f"Best parameters: {grid_search.best_params_}")

        # Evaluate on validation set (if exists)
        if X_val:
             accuracy = self.model.score(X_val, y_val)
             print(f"Validation accuracy: {accuracy:.4f}")
        else:
             print("No separate validation set used.")


        # Cross-validation score on the full augmented dataset
        if len(X_filtered) >= 5: # Need at least 5 samples for 5-fold CV
            cv_scores = cross_val_score(self.model, X_filtered, y_filtered, cv=min(5, len(X_filtered))) # Use fewer folds if data is small
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
        else:
             print("Not enough data for cross-validation.")


        # Feature importance analysis (more complex with FeatureUnion)
        # The original method only considered TF-IDF. We might skip this or
        # try to find a way to show importance for linguistic features too (harder).
        # self._analyze_important_features() # Keep the original method, it might still provide some insight

        # Return validation accuracy or mean CV accuracy
        return accuracy if X_val else (cv_scores.mean() if len(X_filtered) >= 5 else None)


    def _augment_data(self, X, y):
        """Generate synthetic data for Kinyarwanda to balance the dataset"""
        X_aug = X.copy()
        y_aug = y.copy()

        # Count samples for each language
        lang_counts = {}
        for lang in y:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        # Find the language with the most samples among the modeled languages
        max_count = 0
        max_lang = None
        for lang in self.languages: # Only consider languages the model handles
            if lang in lang_counts and lang_counts[lang] > max_count:
                max_count = lang_counts[lang]
                max_lang = lang


        print(f"Data distribution before augmentation: {lang_counts}")

        # Augment Kinyarwanda data by creating variations
        kiny_indices = [i for i, label in enumerate(y) if label == 'kinyarwanda']
        kiny_texts = [X[i] for i in kiny_indices]

        # Calculate how many more samples we need for Kinyarwanda
        kiny_count = lang_counts.get('kinyarwanda', 0)
        # Target count: Aim to match the max language count, or at least double current kiny count if max is low
        target_count = max(max_count, kiny_count * 2 if kiny_count > 0 else 0)
        needed = target_count - kiny_count

        # Simple augmentation (character mixing)
        if needed > 0 and kiny_texts:
            print(f"Augmenting Kinyarwanda data with {needed} synthetic samples (simple mixing)")

            # Generate synthetic samples by combining parts of existing samples
            for _ in range(needed):
                if len(kiny_texts) >= 2:
                    # Select two random texts
                    idx1, idx2 = np.random.choice(len(kiny_texts), 2, replace=False)
                    text1, text2 = kiny_texts[idx1], kiny_texts[idx2]

                    # Ensure texts are long enough to split meaningfully
                    if len(text1) > 5 and len(text2) > 5:
                        # Create a new sample by combining parts
                        split_point1 = len(text1) // random.randint(2, 4) # Split at varying points
                        split_point2 = len(text2) // random.randint(2, 4)

                        # Mix and match parts
                        new_text = text1[:split_point1] + " " + text2[split_point2:] # Add a space

                        # Add small character variations (optional and can introduce noise)
                        # chars = list(new_text)
                        # if len(chars) > 10:
                        #    swap_idx = np.random.randint(0, len(chars)-2)
                        #    chars[swap_idx], chars[swap_idx+1] = chars[swap_idx+1], chars[swap_idx]
                        # new_text = ''.join(chars)

                        X_aug.append(new_text)
                        y_aug.append('kinyarwanda')
                elif kiny_texts: # If only one kiny text, duplicate it (less ideal)
                     X_aug.append(kiny_texts[0])
                     y_aug.append('kinyarwanda')


        # Count samples for each language after augmentation
        lang_counts_after = {}
        for lang in y_aug:
            lang_counts_after[lang] = lang_counts_after.get(lang, 0) + 1

        print(f"Data distribution after augmentation: {lang_counts_after}")

        return X_aug, y_aug

    # Keeping the original _analyze_important_features, though its usefulness
    # for the linguistic features part of the pipeline is limited.
    def _analyze_important_features(self):
        """Analyze and print the most important features for Kinyarwanda detection"""
        if not self.model or not hasattr(self.model, 'named_steps'):
            return

        try:
            # Navigate into the FeatureUnion and then the TF-IDF transformer
            feature_union = self.model.named_steps['features']
            # Find the FeatureUnion step within the pipeline
            if isinstance(feature_union, FeatureUnion):
                 # The transformer_list is inside the FeatureUnion instance
                 transformers = feature_union.transformer_list
                 vectorizer = None
                 # Find the TF-IDF transformer by name or type
                 for name, transformer in transformers:
                     if name == 'tfidf' and isinstance(transformer, TfidfVectorizer):
                         vectorizer = transformer
                         break
                 if vectorizer is None:
                      print("Could not find TF-IDF vectorizer in the pipeline.")
                      return

                 feature_names = vectorizer.get_feature_names_out()

                 # For Naive Bayes in the ensemble (as it's easy to inspect)
                 if hasattr(self.model.named_steps['classifier'], 'named_estimators_'):
                     # The actual classifier estimators are in named_estimators_ after fit
                     ensemble_estimators = self.model.named_steps['classifier'].named_estimators_
                     if 'nb' in ensemble_estimators:
                         nb = ensemble_estimators['nb']
                         if hasattr(nb, 'feature_log_prob_'):
                             # MNBN feature_log_prob_ shape is (n_classes, n_features)
                             # The features correspond to the *output of the FeatureUnion*
                             # which concatenates TF-IDF and Linguistic Features.
                             # We need to know the split point.
                             tfidf_feature_count = len(feature_names)
                             linguistic_feature_count = self.linguistic_features.transform(["test"]).shape[1] # Get number of linguistic features

                             # Get Kinyarwanda class index from the NB model
                             try:
                                class_idx = list(nb.classes_).index('kinyarwanda')
                             except ValueError:
                                print("Kinyarwanda class not found in Naive Bayes model classes.")
                                return

                             # Get log probabilities for Kinyarwanda class
                             feature_importance = nb.feature_log_prob_[class_idx]

                             # Separate importance for TF-IDF and Linguistic Features
                             tfidf_importance = feature_importance[:tfidf_feature_count]
                             linguistic_importance = feature_importance[tfidf_feature_count:]

                             # Get top TF-IDF features
                             if tfidf_feature_count > 0:
                                 top_indices = np.argsort(tfidf_importance)[-20:]
                                 print("\nTop TF-IDF features for Kinyarwanda detection (from NB):")
                                 for idx in top_indices:
                                     print(f"  '{feature_names[idx]}' (log_prob: {tfidf_importance[idx]:.4f})")

                             # We can't easily get meaningful names for linguistic features
                             # unless we explicitly name them in the KinyarwandaLinguisticFeatures transform.
                             # We can at least show their raw importance scores.
                             if linguistic_feature_count > 0:
                                 print("\nLinguistic feature importance scores (from NB):")
                                 # Display scores for linguistic features, perhaps map to descriptions if possible
                                 # Assuming the order matches the transform method's output
                                 linguistic_feature_names_placeholder = [f"Linguistic Feature {j+1}" for j in range(linguistic_feature_count)]
                                 for j, score in enumerate(linguistic_importance):
                                     print(f"  {linguistic_feature_names_placeholder[j]}: {score:.4f}")
                         else:
                             print("Naive Bayes model does not have feature_log_prob_ attribute.")
                     else:
                         print("Naive Bayes estimator 'nb' not found in VotingClassifier.")
            else:
                print("Features step is not a FeatureUnion.")

        except Exception as e:
            print(f"Could not analyze feature importance: {e}")


    def detect(self, text, threshold=0.85):
        """
        Detect if the given text is in Kinyarwanda with advanced confidence scoring
        that incorporates linguistic features and handles code-mixed text.

        Parameters:
        text: string to detect
        threshold: confidence threshold for Kinyarwanda classification (default 0.85)

        Returns:
        dict with language prediction, confidence score, is_kinyarwanda flag,
        and code-mixing information.
        """
        if not self.model:
            raise ValueError("Model not trained. Call train() first.")

        # Basic check for short/empty text
        if not text or len(text.strip()) < 3:
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'is_kinyarwanda': False,
                'code_mixed': False,
                'mix_info': {}
            }

        # Preprocess the text
        processed_text = self.preprocess_text(text)
        words = re.findall(r'\b\w+\b', processed_text) # Need words for some analysis

        # --- Perform Linguistic Analysis using KinyarwandaLinguisticFeatures ---
        # We perform analysis here once and pass results to subsequent steps
        is_code_mixed, primary_lang_analysis, mix_stats = self.linguistic_features._detect_code_mixing(processed_text)
        grammar_score = self.linguistic_features._analyze_grammar_structure(words)
        has_kiny_grammar_foreign = self.linguistic_features._has_kiny_grammar_with_foreign_words(processed_text)
        consonant_violations = self.linguistic_features._check_consecutive_consonants(processed_text)
        # ... (get other relevant scores if needed for boosting/ensemble) ...


        # Apply the core ensemble detection
        # Pass linguistic analysis results to ensemble method if it needs them
        result = self._ensemble_detection(processed_text, grammar_score=grammar_score) # Example: pass grammar score

        # --- Adjust detection based on Code-Mixing Analysis ---
        # Use the results from self.linguistic_features._detect_code_mixing
        if is_code_mixed:
            # If text is detected as mixed, potentially adjust the threshold
            actual_threshold = self.mixed_text_threshold
            # If Kinyarwanda grammatical markers are significantly present in mixed text,
            # boost Kinyarwanda classification possibility.
            if mix_stats.get('kinyarwanda_markers', 0) >= 2:
                 # If the ensemble model didn't call it Kinyarwanda, but linguistic features suggest it...
                 if result['language'] != 'kinyarwanda' and mix_stats.get('kinyarwanda_markers', 0) >= 3:
                     result['language'] = 'kinyarwanda'
                     # Assign a baseline confidence for mixed Kinyarwanda
                     result['confidence'] = max(result['confidence'], 0.7)
                 # If the ensemble model *did* call it Kinyarwanda, boost confidence slightly for confirmed mix+grammar
                 elif result['language'] == 'kinyarwanda':
                     result['confidence'] = min(0.98, result['confidence'] + 0.1) # Small boost

        else:
            # Use the standard threshold if not detected as code-mixed
            actual_threshold = threshold

        # Apply post-detection confidence boosting for Kinyarwanda
        if result['language'] == 'kinyarwanda':
            # Pass all relevant linguistic analysis results to the boosting method
            result['confidence'] = self._boost_kinyarwanda_confidence(
                processed_text, result['confidence'],
                grammar_score=grammar_score,
                has_kiny_grammar_foreign=has_kiny_grammar_foreign,
                consonant_violations=consonant_violations
                # ... add other scores/flags from analysis here ...
            )

        # Apply final threshold using the potentially adjusted actual_threshold
        if result['language'] == 'kinyarwanda' and result['confidence'] >= actual_threshold:
            result['is_kinyarwanda'] = True
        else:
            result['is_kinyarwanda'] = False

        # Add code-mixing info to result
        result['code_mixed'] = is_code_mixed
        if is_code_mixed:
             result['mix_info'] = mix_stats

        # Add other specific analysis results if helpful for debugging or information
        result['analysis_scores'] = {
            'grammar_score': grammar_score,
            'has_kiny_grammar_foreign': has_kiny_grammar_foreign,
            'consonant_violations': consonant_violations,
            # ... include other scores you want to see ...
        }

        return result

    def _ensemble_detection(self, text, grammar_score=0.0): # Added grammar_score parameter
        """
        Apply multiple detection approaches and combine results.
        Incorporates linguistic analysis results passed as parameters.
        """
        # Main model prediction using the pipeline (TF-IDF + Linguistic Features)
        # The .transform step of the pipeline handles FeatureUnion
        proba = self.model.predict_proba([text])[0]

        # Get the highest probability and corresponding language from the model
        max_idx = np.argmax(proba)
        language = self.model.classes_[max_idx]
        confidence = proba[max_idx]

        # Get Kinyarwanda confidence specifically from the model
        kiny_idx = -1
        try:
            kiny_idx = list(self.model.classes_).index('kinyarwanda')
        except ValueError:
            # Should not happen if 'kinyarwanda' is in self.languages and training data
            pass

        kiny_confidence = proba[kiny_idx] if kiny_idx >= 0 else 0.0

        # If another language has high confidence, reduce chance of Kinyarwanda false positive
        # CONSIDER: Only reduce if linguistic analysis *doesn't* strongly support Kinyarwanda
        for i, lang in enumerate(self.model.classes_):
            if lang != 'kinyarwanda' and proba[i] > 0.7:
                 # Reduce less if grammar score is high, suggesting mixed Kinyarwanda structure
                 reduction_factor = 0.95 if grammar_score > 0.5 else 0.9
                 kiny_confidence *= reduction_factor


        # Note: _analyze_linguistic_features is now less crucial here as transform is part of pipeline
        # and specific analysis results are obtained in detect().
        # If _analyze_linguistic_features was meant to provide an *additional* score not in the vector,
        # it would need to be updated in KinyarwandaLinguisticFeatures.
        # Let's remove the call to _analyze_linguistic_features from here as transform is used.
        # The boosting logic will handle incorporating linguistic signals.

        # The characteristic n-gram check is still a separate heuristic layer
        # You might decide to move this into KinyarwandaLinguisticFeatures as well,
        # or keep it here as a specific boosting mechanism. Let's keep it here for now.
        kiny_ngram_boost = self._check_characteristic_ngrams(text)

        # If we detect strong characteristic n-grams, boost Kinyarwanda confidence
        if kiny_ngram_boost > 0:
            kiny_confidence = min(1.0, kiny_confidence + kiny_ngram_boost * 0.05) # Reduced boost amount

        # Update overall result if Kinyarwanda confidence changed significantly
        # (This prioritizes Kinyarwanda if the ensemble + heuristics boost it past the original max)
        if kiny_confidence > confidence and kiny_idx >= 0:
            language = 'kinyarwanda'
            confidence = kiny_confidence
        # Also, if the original prediction was Kinyarwanda, ensure the confidence is the adjusted one
        elif language == 'kinyarwanda':
             confidence = kiny_confidence


        return {
            'language': language,
            'confidence': float(confidence)
        }

    # Removed _analyze_linguistic_features as transform is in the pipeline and
    # specific analyses are called in detect().

    # Keeping _check_characteristic_ngrams as a specific heuristic for boosting
    def _check_characteristic_ngrams(self, text):
        """Check for characteristic Kinyarwanda n-grams (still a heuristic layer)"""
        # Use distinctive Kinyarwanda character patterns from the linguistic features class
        # Access the patterns via the instantiated linguistic_features object
        # Note: This method counts raw pattern occurrences, separate from the FeatureUnion.
        kiny_patterns = self.linguistic_features.distinctive_ibihekane_list # Use a relevant list

        score = 0
        text_lower = text.lower()
        for pattern in kiny_patterns:
            if pattern in text_lower:
                score += text_lower.count(pattern) * 0.5 # Weight occurrences

        return min(5.0, score)  # Cap the boost


    def _boost_kinyarwanda_confidence(self, text, base_confidence, grammar_score=0.0, has_kiny_grammar_foreign=False, consonant_violations=0):
        """
        Apply additional heuristics to boost Kinyarwanda confidence
        using analysis results passed as parameters.
        """
        boost = 0.0

        # Length-based confidence adjustment (remains here)
        if len(text) >= 15:
            boost += 0.03 # Smaller boost

        # Presence of multiple distinctive patterns (relying on _check_characteristic_ngrams indirectly or recalculating)
        # Let's recalculate simple pattern count or use a dedicated check from LF
        # Using a simple count based on a few key patterns
        simple_pattern_count = sum(1 for p in ['nda', 'cy', 'rwa', 'mw', 'by', 'nny', 'shyw'] if p in text.lower())
        if simple_pattern_count >= 2:
            boost += 0.03 * simple_pattern_count

        # Vowel patterns typical in Kinyarwanda (move check to LF or use LF's result if available)
        # Let's add a method in LF for this and call it. Or simply re-do the regex check here for simplicity.
        # Doing the regex check here to keep LF transform output clean, but ideally LF would provide this score
        vowel_pattern_count = len(re.findall(r'[aeiou]{2,}', text.lower()))
        if vowel_pattern_count >= 2:
            boost += 0.02 # Small boost

        # Prefix patterns common in Kinyarwanda (move check to LF or use LF's result)
        # LF's transform includes a prefix score. Could use that, or a simple check here.
        # Using a simple check here based on startsWith
        prefix_patterns_starts = ['umu', 'aba', 'iki', 'ibi', 'uru', 'ama', 'aga', 'icy', 'iby', 'in']
        words = re.findall(r'\b\w+\b', text.lower())
        if any(word.startswith(p) for word in words for p in prefix_patterns_starts):
            boost += 0.04 # Small boost

        # Check for accented vowels (strong indicator of Kinyarwanda) (move check to LF or use LF's result)
        # LF has accented_vowels_pattern. Let's use that.
        accented_vowel_present = bool(self.linguistic_features.accented_vowels_pattern.search(text))
        if accented_vowel_present:
            boost += 0.08 # Significant boost

        # Apply linguistic rules check (using the result passed in)
        # We already got consonant_violations in detect()
        if consonant_violations == 0: # No violations found
             boost += 0.05 # Boost for compliance

        # Use grammar structure score (passed in)
        boost += grammar_score * 0.1 # Add a boost proportional to grammar score

        # Use has_kiny_grammar_foreign flag (passed in)
        if has_kiny_grammar_foreign:
             boost += 0.07 # Boost if mixed grammar with foreign words detected

        # Calculate final confidence with boost (capped at 0.98)
        final_confidence = min(0.98, base_confidence + boost)

        return final_confidence

    # Removed _check_consonant_rule_compliance as it's now in KinyarwandaLinguisticFeatures

    def save_model(self, filepath):
        """Save the trained model to a file"""
        if not self.model:
            raise ValueError("No trained model to save")
        try:
            joblib.dump(self.model, filepath)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model to {filepath}: {e}")

    def load_model(self, filepath):
        """Load a trained model from a file"""
        try:
            self.model = joblib.load(filepath)
            print(f"Model loaded from {filepath}")
            # After loading, re-instantiate the linguistic features?
            # No, the pipeline saves the instance. Ensure the class definition
            # is available when loading.
            # However, if the class definition changes, loaded models might break.
        except FileNotFoundError:
            print(f"Model file not found at {filepath}")
            self.model = None
        except Exception as e:
            print(f"Error loading model from {filepath}: {e}")
            self.model = None


# Example usage in a main function
# (Requires the refactored KinyarwandaLinguisticFeatures class definition above this)
def main():
    # Sample training data (simplified for demonstration)
    # (Keep your existing comprehensive sample data)
    training_data = {
        'kinyarwanda': [
            "ubu umbona ndi kuri move",
            "ntabwo wambona ntakora testing",
            "Muraho, amakuru yanyu?",
            "Ndagukunda cyane.",
            "Mwaramutse neza?",
            "Mwiriwe amahoro?",
            "Umunsi mwiza.",
            "Umunyarwanda yumva ururimi rwe neza.",
            "Igihugu cyacu ni kiza.",
            "Tuzakomeza gutera imbere.",
            "Amafaranga menshi azaza.",
            "Ibyo uvuze nibyo.",
            "Ndabyumva neza cyane.",
            "Ntabwo numva icyo uvuze.",
            "Ibiryo biryoshye cyane.",
            "Nzaza ejo mu gitondo.",
            "Ubuzima bwiza ni ingenzi.",
            "Abana biga neza.",
            "Imana ibane nawe.",
            "Twese turi kumwe.",
            "Ndaseka cyane.",
            "Turaza guhura ryari?",
             # More complex sentences and phrases
            "Amazi ni ingenzi ku buzima bwacu bwose.",
            "Umukino utangira saa tatu z'amanywa.",
            "Umuryango wanjye ni mwiza cyane kandi tubanye neza.",
            "Ikiganiro cyacu kiragenda neza uyu munsi.",
            "Ndagukumbura cyane kuva aho ugendeye",
            "Ndakwishimira cyane kubera ibyo wakoze byose.",
            "Amashuri yigisha neza abana bacu.",
            "Umurima wacu urera neza.",
            "Nzava mu modoka vuba cyane.",
            "Ndasoma igitabo cyiza cyanditswe n'umwanditsi w'umunyarwanda.",

            # Additional phrases covering various contexts
            "Abandi barimo gukora iki muri iki gihe?",
            "Nkora mu ruganda runini rw'iKigali.",
            "Ndabyumva ariko ntabwo nshobora kubisobanura neza.",
            "Ntabwo nabashije kubona imodoka yanjye.",
            "Urukundo rwawe ni rwinshi cyane.",
            "Abana bacu biga mu ishuri ryiza.",
            "Umwarimu afasha abanyeshuri benshi.",
            "Ndashaka kumenya uko wageze aha.",
            "Igihe kirageze ngo tugende.",
            "Dukeneye gutegura ibiryo byinshi.",

            # Including some words the model missed
            "Abandi baraho se badi?",
            "Ndashaka kumenya igihe",
            "Ubu ndimo kwiga Ikinyarwanda",
            "Abanyarwanda benshi bavuga Ikinyarwanda neza",
            "Ndaza kugaruka vuba",
            "Amaherezo twabonye umuti w'ikibazo",
            "Yego ndabyemera rwose",
            "Ntabwo numva icyo uri kuvuga",
            "Mu by'ukuri nshoshoboye gukora ibi",
            "Ese uzaza i Kigali ryari?",
            "Mu Rwanda hari imisozi igihumbi",

            # Adding the examples that caused false negatives in our test
            "Abandi baraho se badi?",

            # Add variations of the problematic phrase
            "Nyabuna se abandi baraho?",
            "Baraho se badi?",
            "Baraho badi?",
            "Baraho se?",
            "Muraho badi?",
            "Abandi baraho?",
            "Baraho nyabuna?",

             # Add some mixed examples to the training data (optional, synthetic)
             # You could generate these using the _generate_code_mixed_examples logic
             # from the first class version, perhaps as a separate utility function.
             # Or manually add them here.
             "Ndi online ubu.", # Kinyarwanda + English
             "Ndashaka ku download application.", # Kinyarwanda + English
             "Ubu ndimo gukora project.", # Kinyarwanda + English
             "Twari muri meeting.", # Kinyarwanda + English
             "Amafaranga ya bank.", # Kinyarwanda + English
             "Ura charging telephone?", # Kinyarwanda + English
             "Turi kuri facebook.", # Kinyarwanda + English
             "Ndamubona kuri whatsapp.", # Kinyarwanda + English
             "Ese ufite message yanjye?", # Kinyarwanda + English
             "Niba uri okay, tuza.", # Kinyarwanda + English
             "Nshaka kujya kuri internet.", # Kinyarwanda + English
             "Bon courage muri school.", # Kinyarwanda + French/English
             "Urakoze sana mwami.", # Kinyarwanda + Swahili/Kinyarwanda (sana is Swahili)
             "Ntabwo nabonye document.", # Kinyarwanda + French
             "Umu bureau muri kigali.", # Kinyarwanda + French
             "Twahuye muri centre.", # Kinyarwanda + French
             "Ibyo ni sawa kabisa.", # Kinyarwanda + Swahili
             "Habari ko muri mu Rwanda?", # Swahili + Kinyarwanda
             "Asante cyane.", # Swahili + Kinyarwanda


        ],
        'english': [
            "Hello, how are you?",
            "The quick brown fox jumps over the lazy dog.",
            "Where are you going?",
            "This is an English sentence.",
            "I can't speak Kinyarwanda.",
            "What time is the meeting?",
            "Thank you for your help.",
            "Can you please repeat that?",
            "Have a nice day!",
            "Let's meet tomorrow morning.",
            "The weather is nice today.",
            "I am learning a new language.",
            "She works at the university.",
            "Can you help me with this?",
            "Tomorrow we will go to the beach.",
            "Please call me later.",
            "What is your favorite food?",
            "The meeting starts at three o'clock.",
            "My family is very important to me.",
            "Our conversation is going well.",
            "I miss you very much.",
            "The school teaches well.",
            "I'll get out of the car.",
            "How was your weekend?",
            "I enjoyed our time together.",
            "Can we schedule another meeting?",
            "The project is almost complete.",
            "I need more information about this topic.",
            "Please send me the documents by email.",
            "The restaurant is on the corner of the street.",
        ],
        'french': [
            "Je ne comprends pas ce que tu dis.",
            "Bonjour tout le monde.",
            "Comment allez-vous aujourd'hui?",
            "Je suis très content de vous voir.",
            "Merci beaucoup pour votre aide.",
            "Je voudrais un café, s'il vous plaît.",
            "Quel temps fait-il aujourd'hui?",
            "Je parle français un peu.",
            "La plume de ma tante est sur le bureau de mon oncle.",
            "Nous allons au cinéma ce soir.",
            "Je travaille à l'université.",
            "C'est une très belle journée.",
            "Je ne sais pas comment expliquer ça.",
            "Avez-vous compris ce que j'ai dit?",
            "Nous devons partir maintenant.",
        ],
        'swahili': [
            "Habari yako?",
            "Jina langu ni Claude.",
            "Ninasema Kiswahili kidogo.",
            "Asante sana kwa msaada wako.",
            "Tutaonana kesho.",
            "Chakula hiki ni kitamu sana.",
            "Naenda sokoni kununua matunda.",
            "Leo ni siku nzuri.",
            "Unaenda wapi sasa?",
            "Nimefurahi kukutana nawe.",
            "Unafanya kazi gani?",
            "Naishi Nairobi.",
            "Ninapenda kusoma vitabu.",
            "Hali ya hewa ni nzuri leo.",
            "Tutaonana baadaye.",
        ],
        'other': [
            "Hola, ¿cómo estás hoy?",  # Spanish
            "Ich bin ein Student.",  # German
            "Ich bin müde und gehe schlafen.", # German
            "No entiendo lo que estás diciendo.", # Spanish
            "Mi chiamo Claude e vengo dall'Italia.", # Italian
            "Dov'è la stazione ferroviaria?", # Italian
            "Jeg snakker ikke norsk.", # Norwegian
            "Talar du svenska?", # Swedish
            "¿Dónde está la biblioteca?", # Spanish
            "Wie spät ist es?", # German
            "Ik spreek een beetje Nederlands.", # Dutch
            "Jeg er meget glad i dag.", # Danish
            "Olá, como vai você?", # Portuguese
            "Não falo português muito bem.", # Portuguese
            "Det er en vakker dag i dag.", # Norwegian
        ]
    }

    # Create and train the enhanced detector
    detector = EnhancedKinyaLangDetector()
    detector.train(training_data)

    # Check if a trained model exists and load it, otherwise train
    # model_filepath = "kinya_detector.joblib"
    # try:
    #     detector.load_model(model_filepath)
    # except FileNotFoundError:
    #     print(f"Model file not found at {model_filepath}. Training new model...")
    #     detector.train(training_data)
    #     detector.save_model(model_filepath)


    # Example usage in a chatbot
    def chatbot_response(user_input):
        result = detector.detect(user_input)

        response = f"Input: '{user_input}'\n"
        response += f"Detection Result: {result}\n"

        if result['is_kinyarwanda']:
            confidence_percent = int(result['confidence'] * 100)
            response += (f"Chatbot interpretation: I detected Kinyarwanda (confidence: {confidence_percent}%). "
                        f"I'm still learning Kinyarwanda, but I'll try my best to understand. "
                        f"Would you like to continue in English for now?")
        elif result['code_mixed']:
             confidence_percent = int(result['confidence'] * 100)
             response += (f"Chatbot interpretation: I detected code-mixed text (confidence: {confidence_percent}%, predicted primary: {result['language']}). "
                         f"It seems to contain Kinyarwanda mixed with other languages. "
                         f"How can I help with this?")
        else:
            # Process the message normally in the detected language
            if result['language'] == 'english':
                response += "Chatbot interpretation: I understood your message in English. How can I help you today?"
            elif result['language'] == 'french':
                response += "Chatbot interpretation: J'ai compris votre message en français. Comment puis-je vous aider aujourd'hui?"
            elif result['language'] == 'swahili':
                response += "Chatbot interpretation: Nimeelewa ujumbe wako kwa Kiswahili. Nawezaje kukusaidia leo?"
            else:
                 response += f"Chatbot interpretation: I detected the language as '{result['language']}' (confidence: {int(result['confidence']*100)}%). How can I help?"

        return response


    # Test some examples
    test_phrases = [
        "Ese ubu yamaze gu testing?", # Mixed Kinyarwanda + English
        "uraho, witwa nde?", # Kinyarwanda
        "Bite byanyu?", # Kinyarwanda greeting
        "habari zenu?", # Swahili greeting
        "Hello, how are you?", # English greeting
        "Wambwira isaha?", # Kinyarwanda
        "Je parle français", # French
        "Ubu ndimo kwiga Ikinyarwanda", # Kinyarwanda
        "Can you help me with my homework?", # English
        "Iyakaremye niyo ikamena", # Kinyarwanda proverb
        "Ibyabapfu biribwa n'abapfumu", # Kinyarwanda proverb
        "Ndi online ubu.", # Mixed Kinyarwanda + English
        "Ura charging telephone?", # Mixed Kinyarwanda + English
        "Twahuye muri centre.", # Mixed Kinyarwanda + French
        "Ibyo ni sawa kabisa.", # Mixed Kinyarwanda + Swahili
        "Mfite message kuri whatsapp.", # Mixed Kinyarwanda + English
        "Murakoze cyane.", # Kinyarwanda (Common phrase)
        "Tuzahura ejo.", # Kinyarwanda
        "Where are you going?", # English
        "Quelle heure est-il?", # French
        "Unataka nini?", # Swahili
        "Amafaranga ntabwo ahagije.", # Kinyarwanda
        "Ndashaka gu checkinga email zanjye.", # Mixed Kinyarwanda + English
        "Mwiriwe neza.", # Kinyarwanda greeting
        "Nta kibazo.", # Kinyarwanda
        "Okay, see you later.", # Mixed English
        "Bon appétit!", # French
        "Hakuna matata.", # Swahili
        "Ese ubu uri gukora project?", # Mixed Kinyarwanda + English
    ]

    for phrase in test_phrases:
        print("-" * 30)
        print(chatbot_response(phrase))


if __name__ == "__main__":
    main()