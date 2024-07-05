import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd
from .additional_data_generator import Additional_data_generator


class Common_words_generator(Additional_data_generator):
    word_list: dict[str, tuple[list, list]]

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

        # Compute word counts
        self._get_word_counts()

    def get_sample(self, metadata: dict) -> str:
        """Get the batch of samples from the list

        :return: The next batch of samples (first element is the data and second is the label).
        """
        words, counts = self.word_list[metadata["label"].lower()]

        # Select words randomly
        sampled_words = random.choices(words, weights=counts, k=self.batch_size)

        # Return the data and label
        return "\n".join(sampled_words)

    @staticmethod
    def get_name() -> str:
        return "common_words_generator"

    def _get_word_counts(self) -> None:
        def preprocess(text: str):
            tokens = word_tokenize(text.lower())  # Tokenize and convert to lower case
            words = [word for word in tokens if word.isalpha()]  # Remove punctuation
            stop_words = set(stopwords.words('english'))
            words = [word for word in words if word not in stop_words]  # Remove stopwords
            return words

        def count_words(group):
            words = []
            for text in group['processed_texts']:
                words.extend(text)

            return Counter(words)

        # Read the CSV files
        data_frames = []
        for csv_location in self.file_paths:
            data_frames.append(pd.read_csv(csv_location))
        df = pd.concat(data_frames, ignore_index=True)

        # Apply preprocessing to the text column
        df['processed_texts'] = df['text'].apply(preprocess)

        # Group by emotion and apply the counting function
        word_counts_by_emotion = df.groupby('label').apply(count_words)

        # FIlter out words that appear less than 2 times and words with sentiment more neutral than 0.2
        most_common_words = {emotion: [el for el in counts.items() if el[1] > 2] for emotion, counts in word_counts_by_emotion.items()}  # noqa

        self.word_list = {}
        for emotion, word_list in most_common_words.items():
            words, counts = zip(*word_list)

            self.word_list[emotion] = (words, counts)
