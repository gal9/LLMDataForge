import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from .additional_data_generator import Additional_data_generator


class Common_emotional_words_generator2(Additional_data_generator):
    """I think this one does not separate on emotions.
    """
    word_list: dict[str, tuple[list, list]]
    file_paths: list[str]
    sia: SentimentIntensityAnalyzer

    def __init__(self) -> None:
        super().__init__()

    def configure(self, config: dict) -> None:
        super().configure(config)

        nltk.download('vader_lexicon')

        self.file_paths = config["data_files"]

        # Set random seed if specified
        if ("seed" in config):
            random.seed(config["seed"])

        self.prompt_placeholder = config.get("prompt_placeholder", "used_word")

        self.sia = SentimentIntensityAnalyzer()

        # Compute word counts
        self._get_word_counts()

    def get_sample(self, metadata: dict) -> str:
        """Get the batch of samples from the list

        :return: The next batch of samples (first element is the data and second is the label).
        """
        words, counts_pos, counts_neg = self.word_list

        label_emotion = self.sia.polarity_scores(metadata["label"])["compound"]

        if (label_emotion > 0):
            counts = counts_pos
        else:
            counts = counts_neg

        # Select words randomly
        sampled_words = random.choices(words, weights=counts, k=self.batch_size)

        # Return the data and label
        return "\n".join(sampled_words)

    @staticmethod
    def get_name() -> str:
        return "common_emotional_words_generator2"

    def _get_word_counts(self) -> None:
        def preprocess(text: str):
            tokens = word_tokenize(text.lower())  # Tokenize and convert to lower case
            words = [word for word in tokens if word.isalpha()]  # Remove punctuation
            stop_words = set(stopwords.words('english'))
            words = [word for word in words if word not in stop_words]  # Remove stopwords
            return words

        # Read the CSV files
        data_frames = []
        for csv_location in self.file_paths:
            data_frames.append(pd.read_csv(csv_location))
        df = pd.concat(data_frames, ignore_index=True)

        # Combine all comments into a single string
        all_comments = ' '.join(df['Comment'])

        # Tokenize and count words
        word_counts = Counter(preprocess(all_comments))

        # FIlter out words that appear less than 2 times and words with sentiment more neutral than 0.2
        most_common_emotional_words = [(el[0], el[1], self.sia.polarity_scores(el[0])["compound"]) for el in word_counts.items() if (el[1] > 2 and abs(self.sia.polarity_scores(el[0])["compound"]) > 0.2)]  # noqa

        # Return word, importance for positive and importance for negative
        self.word_list = [x[0] for x in most_common_emotional_words], [x[1] if x[2] > 0 else x[1]/abs(x[2]*10) for x in most_common_emotional_words], [x[1] if x[2] < 0 else x[1]/abs(x[2]*10) for x in most_common_emotional_words] # noqa
