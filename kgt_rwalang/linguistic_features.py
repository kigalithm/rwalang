import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from kgt_rwalang.config import (
    EN_LOAN_WORDS,
    FR_LOAN_WORDS,
    SW_LOAN_WORDS,
    KINYA_MARKERS,
    KINYA_AFFIX_SUFFIXES,
    KINYA_AFFIX_PREFIXES,
    IBIHEKANE,
    IBYUNGO,
    INGOMBAJWI,
    DISTINCTIVE_IBIHEKANE,
    COMMON_KINYA_PATTERNS,
)


class KinyaLinguisticFeatures(BaseEstimator, TransformerMixin):
    """
    Extract Kinyarwanda-specific linguistic features and perform linguistic analysis from text.

    This transformer analyzes text for the presence of specific Kinyarwanda linguistic
    elements like ibihekane (consonant clusters), inyajwi (vowels with accents),
    ibyungo (connectives), specific phonological patterns, grammatical markers,
    affixes, loan words, and code-mixing characteristics.
    """

    def __init__(self):
        # Initialize Kinyarwanda-specific linguistic components and data

        self.loan_words = {
            "english":EN_LOAN_WORDS,
            "french": FR_LOAN_WORDS,
            "swahili": SW_LOAN_WORDS,
        }

        self.kiny_markers = KINYA_MARKERS

        self.kiny_affixes = {
            "prefixes": KINYA_AFFIX_SUFFIXES,  # Expanded list
            "suffixes": KINYA_AFFIX_PREFIXES,  # Expanded list
        }

        # Initialize Kinyarwanda-specific linguistic components
        self.ibihekane = IBIHEKANE

        self.ibyungo = IBYUNGO

        self.inyajwi = ["a", "e", "i", "o", "u", "â", "ê", "î", "ô", "û"]
        self.accented_vowels = ["â", "ê", "î", "ô", "û"]
        self.regular_vowels = ["a", "e", "i", "o", "u"]

        self.ingombajwi = INGOMBAJWI

        self.indomo = ["u", "a", "i"]  # Labial vowels related to harmony

        # Compile regex patterns for efficient matching
        self.ibihekane_pattern = self._compile_regex_pattern(self.ibihekane)
        self.distinctive_ibihekane_list = DISTINCTIVE_IBIHEKANE

        self.distinctive_ibihekane_pattern = self._compile_regex_pattern(
            self.distinctive_ibihekane_list
        )

        self.ibyungo_pattern = self._compile_word_boundary_pattern(self.ibyungo)
        self.accented_vowels_pattern = re.compile(f"[{''.join(self.accented_vowels)}]")

        # Compile regex for Kinyarwanda markers and affixes for efficiency
        self.kiny_markers_pattern = self._compile_word_boundary_pattern(
            self.kiny_markers
        )
        self.kiny_prefixes_pattern = self._compile_regex_start_pattern(
            self.kiny_affixes["prefixes"]
        )
        self.kiny_suffixes_pattern = self._compile_regex_end_pattern(
            self.kiny_affixes["suffixes"]
        )

        # Compile regex for loan words (can be extensive, handle carefully)
        all_loan_words = [
            word for sublist in self.loan_words.values() for word in sublist
        ]
        self.loan_words_pattern = self._compile_word_boundary_pattern(all_loan_words)

    def _compile_regex_pattern(self, items):
        """Compile a regex pattern to find any of the items"""
        # Sort by length (longest first) to ensure proper matching
        sorted_items = sorted(items, key=len, reverse=True)
        pattern = "|".join(map(re.escape, sorted_items))
        return re.compile(pattern, re.IGNORECASE)

    def _compile_word_boundary_pattern(self, items):
        """Compile a regex pattern to find any of the items with word boundaries"""
        # Sort by length (longest first) to ensure proper matching
        sorted_items = sorted(items, key=len, reverse=True)
        pattern = "|".join(rf"\b{re.escape(item)}\b" for item in sorted_items)
        return re.compile(pattern, re.IGNORECASE)

    def _compile_regex_start_pattern(self, items):
        """Compile a regex pattern to find any of the items at the start of a word"""
        # Sort by length (longest first)
        sorted_items = sorted(items, key=len, reverse=True)
        pattern = "|".join(rf"\b{re.escape(item)}\w*" for item in sorted_items)
        return re.compile(pattern, re.IGNORECASE)

    def _compile_regex_end_pattern(self, items):
        """Compile a regex pattern to find any of the items at the end of a word"""
        # Sort by length (longest first)
        sorted_items = sorted(items, key=len, reverse=True)
        pattern = "|".join(rf"\w*{re.escape(item)}\b" for item in sorted_items)
        return re.compile(pattern, re.IGNORECASE)

    def fit(self, X, y=None):
        """Nothing to fit for this transformer"""
        return self

    def transform(self, X):
        """Extract Kinyarwanda-specific linguistic features from texts"""
        if isinstance(X, str):
            X = [X]  # Convert single string to list

        # Number of features calculated by transform
        # Let's estimate: ibihekane density, distinctive ibihekane density, ibyungo per word,
        # accented vowel ratio, vowel-consonant ratio, consonant rule compliance,
        # vowel harmony score, prefix pattern score, marker count, loan word count,
        # grammar structure score, kiny patterns count
        num_features = 12  # Initial estimate, adjust as features are added below

        features = np.zeros((len(X), num_features))

        for i, text in enumerate(X):
            # Skip empty text
            if not text or len(text.strip()) == 0:
                continue

            # Normalize text to lowercase
            processed_text = text.lower()
            words = re.findall(r"\b\w+\b", processed_text)
            text_length = len(processed_text)
            word_count = len(words)

            if word_count == 0:  # Avoid division by zero
                continue

            # 1. Ibihekane (consonant clusters) features
            ibihekane_matches = self.ibihekane_pattern.findall(processed_text)
            ibihekane_count = len(ibihekane_matches)

            # 2. Distinctive ibihekane (more unique to Kinyarwanda)
            distinctive_matches = self.distinctive_ibihekane_pattern.findall(
                processed_text
            )
            distinctive_count = len(distinctive_matches)

            # 3. Ibyungo (connectives) features
            ibyungo_matches = self.ibyungo_pattern.findall(processed_text)
            ibyungo_count = len(ibyungo_matches)

            # 4. Accented vowels features
            accented_vowels_count = len(
                self.accented_vowels_pattern.findall(processed_text)
            )
            all_vowels_count = sum(processed_text.count(v) for v in self.inyajwi)

            # 5. Consonant pattern features
            consonants_count = sum(processed_text.count(c) for c in self.ingombajwi)

            # 6. Check for consecutive consonant rule compliance (0 or 1)
            consecutive_violations = self._check_consecutive_consonants(processed_text)

            # 7. Check for vowel harmony patterns (score)
            vowel_harmony_score = self._check_vowel_harmony(processed_text)

            # 8. Check for common Kinyarwanda prefixes (score)
            prefix_pattern_score = self._check_prefix_patterns(processed_text)

            # --- Features derived from Moved Data/Methods ---

            # 9. Count Kinyarwanda markers
            kiny_marker_count = len(self.kiny_markers_pattern.findall(processed_text))

            # 10. Count Loan words
            loan_word_count = len(self.loan_words_pattern.findall(processed_text))

            # 11. Analyze Grammar Structure (score)
            grammar_structure_score = self._analyze_grammar_structure(words)

            # 12. Count characteristic Kinyarwanda patterns within words
            kiny_pattern_word_count = sum(
                1 for word in words if self._has_kiny_patterns(word)
            )
            kiny_patterns_score = (
                kiny_pattern_word_count / word_count
            )  # Ratio of words with kiny patterns

            # Calculate normalized feature values
            features[i, 0] = ibihekane_count / max(10, text_length) * 10
            features[i, 1] = distinctive_count / max(5, text_length) * 20
            features[i, 2] = ibyungo_count / word_count
            features[i, 3] = accented_vowels_count / max(1, all_vowels_count)
            features[i, 4] = all_vowels_count / max(1, consonants_count)
            features[i, 5] = (
                1.0 if consecutive_violations == 0 else 0.0
            )  # Changed from 0.5 to 0.0 for clearer binary feature
            features[i, 6] = vowel_harmony_score
            features[i, 7] = prefix_pattern_score
            features[i, 8] = kiny_marker_count / word_count  # Marker density
            features[i, 9] = loan_word_count / word_count  # Loan word density
            features[i, 10] = grammar_structure_score
            features[i, 11] = kiny_patterns_score

        return features

    # --- Linguistic Analysis Methods (Moved from EnhancedKinyaLangDetector) ---

    def _has_kiny_patterns(self, word):
        """Check if a word contains characteristic Kinyarwanda patterns (moved)"""
        # Distinctive Kinyarwanda character patterns (expanded for better coverage)
        kiny_patterns = COMMON_KINYA_PATTERNS

        # Use a compiled regex pattern for efficiency if needed, but for a single word,
        # a simple loop is fine for now.
        return any(pattern in word.lower() for pattern in kiny_patterns)

    def _analyze_grammar_structure(self, words):
        """
        Analyze if the text follows Kinyarwanda grammatical patterns (moved)
        Returns a score between 0 and 1
        """
        if not words:
            return 0.0

        score = 0.0
        total_checks = 0

        # Use the stored markers and affixes

        # 1. Check for question markers at beginning or end
        total_checks += 1
        if words[0].lower() in ["ese", "mbese"] or (
            len(words) > 0 and words[-1].lower() in ["se", "ko", "bite"]
        ):
            score += 1

        # 2. Check for subject prefixes followed by verb markers/roots
        # This is complex and approximate without proper parsing.
        # A simplified check: look for words starting with a subject prefix followed by a common verb marker or pattern.
        prefix_verb_patterns = ["n", "u", "a", "tu", "mu", "ba"]
        common_verb_starts = [
            "ra",
            "za",
            "ri",
            "ga",
            "ye",
            "ko",
            "kug",
            "guk",
            "kwib",
        ]  # common verb forms/markers
        found_prefix_verb = False
        total_checks += 1  # One check for this pattern type
        for word in words:
            lower_word = word.lower()
            for prefix in prefix_verb_patterns:
                if lower_word.startswith(prefix) and len(lower_word) > len(prefix):
                    remaining = lower_word[len(prefix) :]
                    if any(
                        remaining.startswith(vb_start)
                        for vb_start in common_verb_starts
                    ):
                        score += 1
                        found_prefix_verb = True
                        break  # Found one instance, sufficient for this check type
            if found_prefix_verb:
                break

        # 3. Check for locative markers at the start of words
        loc_markers_starts = ["mu", "ku", "i", "muri", "kuri", "aha"]
        found_locative = False
        total_checks += 1  # One check for this pattern type
        for word in words:
            lower_word = word.lower()
            if any(
                lower_word.startswith(loc) and len(lower_word) > len(loc)
                for loc in loc_markers_starts
            ):
                score += 1
                found_locative = True
                break
        if found_locative:
            pass  # Added to satisfy structure

        # 4. Check for negative constructions (nta, nti + pronoun prefix)
        negative_patterns = ["nta", "nti"]  # nti is followed by pronoun prefix
        found_negative = False
        total_checks += 1
        for word in words:
            lower_word = word.lower()
            if lower_word.startswith("nta") or (
                lower_word.startswith("nti")
                and len(lower_word) > 3
                and lower_word[3] in ["n", "u", "a", "tu", "mu", "ba"]
            ):
                score += 1
                found_negative = True
                break
        if found_negative:
            pass  # Added to satisfy structure

        # 5. Check for possessive constructions (word starting with possessive marker)
        # This is also approximate. Check for words starting with common possessive markers followed by something else.
        poss_markers_starts = ["wa", "ba", "ya", "za", "cya", "bya", "rya", "ka"]
        found_possessive = False
        total_checks += 1
        for word in words:
            lower_word = word.lower()
            if any(
                lower_word.startswith(pm) and len(lower_word) > len(pm)
                for pm in poss_markers_starts
            ):
                score += 1
                found_possessive = True
                break
        if found_possessive:
            pass  # Added to satisfy structure

        # Normalize score by number of checks
        return min(1.0, score / total_checks if total_checks > 0 else 0.0)

    def _detect_code_mixing(self, text):
        """
        Detect if text contains mixed language elements (moved)
        Returns: (is_mixed, primary_language, statistics)
        """
        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) < 2:
            return False, None, {}

        # Count words from each language
        lang_counts = {
            "kinyarwanda": 0,
            "english": 0,
            "french": 0,
            "swahili": 0,
            "other": 0,
        }
        loan_word_counts = {lang: 0 for lang in self.loan_words.keys()}

        # Track Kinyarwanda grammatical markers and affixes
        kiny_marker_count = 0
        kiny_affix_word_count = 0

        # Analyze each word
        for word in words:
            word_is_loan = False
            # Check for loan words first
            for lang, loan_list in self.loan_words.items():
                if word in loan_list:
                    loan_word_counts[lang] += 1
                    lang_counts[
                        lang
                    ] += 1  # Count loan words towards their origin language count
                    word_is_loan = True
                    break

            if word_is_loan:
                continue

            # If not a recognized loan word, check for Kinyarwanda indicators
            # Check for grammatical markers
            if word in self.kiny_markers:
                kiny_marker_count += 1
                lang_counts["kinyarwanda"] += 1  # Count markers as Kinyarwanda
                continue

            # Check for Kinyarwanda affixes (prefixes/suffixes)
            has_kiny_affix = False
            if self.kiny_prefixes_pattern.search(
                word
            ) or self.kiny_suffixes_pattern.search(word):
                kiny_affix_word_count += 1
                lang_counts[
                    "kinyarwanda"
                ] += 1  # Count words with Kinyarwanda affixes as Kinyarwanda
                has_kiny_affix = True

            if has_kiny_affix:
                continue

            # Check for characteristic Kinyarwanda patterns (ibihekane-like within words)
            if self._has_kiny_patterns(word):
                # Only count this towards Kinyarwanda if it wasn't already counted by markers or affixes
                lang_counts["kinyarwanda"] += 1
                continue

            # If the word didn't match any specific Kinyarwanda or loan patterns,
            # a more general language detector or word list check would be needed here
            # to assign it to 'english', 'french', 'swahili', or 'other'.
            # For simplicity in this refactoring, we'll just count it towards 'other' if not matched.
            # In a real system, you'd integrate a general word-level language ID here.
            # For now, the logic is based on the provided lists.

        # Determine if code-mixed
        total_loan_words = sum(loan_word_counts.values())
        total_words = len(words)

        # Calculate total count of explicit Kinyarwanda indicators (markers + words with affixes)
        total_kiny_indicators = kiny_marker_count + kiny_affix_word_count

        # Calculate primary language based on counts from known lists
        primary_lang = "other"
        max_count = 0
        for lang, count in lang_counts.items():
            if count > max_count:
                max_count = count
                primary_lang = lang

        # Define mixed criteria:
        # 1. Presence of Kinyarwanda indicators AND loan words
        # 2. Kinyarwanda is the primary language AND a significant portion of words are loans
        # 3. A significant mix of words from multiple languages according to counts
        # This is a heuristic and can be tuned.
        is_mixed = (
            (total_kiny_indicators >= 1 and total_loan_words >= 1)
            or (primary_lang == "kinyarwanda" and total_loan_words / total_words > 0.15)
            or (
                sum(1 for count in lang_counts.values() if count > 0) > 1
                and total_words > 5
            )
        )  # More than one language detected with >0 words in a longer text

        stats = {
            "word_count": total_words,
            "kinyarwanda_markers": kiny_marker_count,
            "kinyarwanda_affix_words": kiny_affix_word_count,
            "loan_words_counts": loan_word_counts,
            "total_loan_words": total_loan_words,
            "language_raw_counts": lang_counts,  # Raw counts from analysis
            "language_distribution": {
                k: v / total_words for k, v in lang_counts.items() if total_words > 0
            },  # Distribution based on our lists
        }

        # Refine primary_lang based on normalized distribution if needed, but raw counts are a start.
        # The primary_lang calculated here is based on word counts from *our defined lists*.
        # The calling class might use its own model prediction for the true primary_lang.

        return is_mixed, primary_lang, stats

    def _has_kiny_grammar_with_foreign_words(self, text):
        """Detect if text has Kinyarwanda grammatical structure with foreign vocabulary (moved)"""
        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) < 3:
            return False

        # Check for key grammatical markers using the precompiled pattern
        has_kiny_markers = bool(self.kiny_markers_pattern.search(text.lower()))

        # Check for foreign words using the precompiled pattern
        has_foreign_words = bool(self.loan_words_pattern.search(text.lower()))

        if not has_kiny_markers or not has_foreign_words:
            return False  # Must have both types of indicators

        # Look for specific code-mixing patterns using the parsed words

        # Pattern 1: "Verb + gu/ku + English verb/loan word"
        # This is complex as Kinyarwanda verbs conjugate. Check for common infinitive + loan word structure.
        english_loan_verbs = self.loan_words["english"]  # Use specific list if possible
        for i in range(len(words) - 1):
            if words[i] in ["gu", "ku"]:  # infinitive marker
                if (
                    words[i + 1] in english_loan_verbs
                ):  # English verb or common loan word used as verb
                    return True

        # Pattern 2: Kinyarwanda interrogative + foreign word
        if words[0] in ["ese", "mbese"]:
            if any(
                self.loan_words_pattern.search(words[i]) for i in range(1, len(words))
            ):
                return True

        # Pattern 3: Kinyarwanda subject prefix + foreign word acting as a verb root
        # This is highly approximate. Check for words starting with a subject prefix followed by a loan word.
        kiny_prefixes = ["n", "u", "a", "tu", "mu", "ba"]
        for word in words:
            lower_word = word.lower()
            for prefix in kiny_prefixes:
                if lower_word.startswith(prefix) and len(lower_word) > len(prefix):
                    remaining = lower_word[len(prefix) :]
                    if remaining in self.loan_words_pattern.findall(
                        remaining
                    ):  # Check if the remaining part is a loan word
                        return True

        # Pattern 4: Kinyarwanda noun prefix + foreign word acting as a noun root
        kiny_noun_prefixes = self.kiny_affixes["prefixes"]  # Use the expanded list
        for word in words:
            lower_word = word.lower()
            for prefix in kiny_noun_prefixes:
                if lower_word.startswith(prefix) and len(lower_word) > len(prefix):
                    remaining = lower_word[len(prefix) :]
                    if remaining in self.loan_words_pattern.findall(remaining):
                        return True

        # If no specific pattern matched, but general indicators (markers + loans) are present
        # This thresholding is a bit redundant if the calling method also does it,
        # but we can return True as a general indicator of mixed grammar/vocab.
        # Let's stick to returning True only if specific patterns or a strong general mix is found.
        # The initial check (has_kiny_markers and has_foreign_words) covers the general mix.

        return (
            has_kiny_markers and has_foreign_words
        )  # Return True if both general types are present
    

    def _check_consecutive_consonants(self, text):
        """
        Check for violations of Kinyarwanda phonological rule:
        'It's prohibited to have consecutive identical consonants except n in nny'
        (Existing method, slightly improved pattern)
        """
        violations = 0

        # Pattern to find consecutive identical consonants (case-insensitive)
        # Excludes 'nn' which is handled as a special case
        pattern = re.compile(r"([bcdfghjklmpqrstvwxyz])\1+", re.IGNORECASE)
        matches = pattern.finditer(text.lower())  # Ensure lowercase for check

        for match in matches:
            matched_string = match.group()
            # The only exception is 'nn' specifically as part of 'nny'
            if matched_string == "nn":
                # Check if this 'nn' is followed by 'y'
                end_pos = match.end()
                if end_pos < len(text) and text.lower()[end_pos] == "y":
                    # This is 'nny', which is allowed
                    continue
            # Any other consecutive identical consonant or 'nn' not followed by 'y' is a violation
            violations += 1

        return violations

    def _check_vowel_harmony(self, text):
        """
        Check for Kinyarwanda vowel harmony patterns, particularly focusing on
        indomo vowels (u, a, i) and their co-occurrence.
        (Existing method, minor adjustment)
        """
        # Count occurrences of indomo vowels in sequence
        indomo_count = 0
        lower_text = text.lower()
        for i in range(len(lower_text) - 1):
            if lower_text[i] in self.indomo and lower_text[i + 1] in self.indomo:
                indomo_count += 1

        # Normalize by text length or number of vowel pairs
        # Normalizing by text length is simpler and less prone to division by zero
        score = indomo_count / max(10, len(lower_text)) * 5  # Scale the score a bit

        return min(1.0, score)  # Cap at 1.0

    def _check_prefix_patterns(self, text):
        """
        Check for common Kinyarwanda prefixes (noun class markers, verb prefixes).
        Returns a score based on the presence of these patterns.
        (Existing method, uses compiled pattern now)
        """
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return 0.0

        score = 0
        # Use the precompiled prefix pattern
        for word in words:
            if self.kiny_prefixes_pattern.search(word):
                score += 1

        # Normalize by word count, capped at 1.0
        return min(1.0, score / len(words))
