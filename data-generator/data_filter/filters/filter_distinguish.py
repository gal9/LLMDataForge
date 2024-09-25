from typing import Dict, List
from .filter import Filter
import pandas as pd
from data_filter.dataset_handlers import Dataset_handler

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import make_pipeline, Pipeline
from nltk.corpus import stopwords


def make_NB_model() -> Pipeline:
    # Creating a Count Vectorizer to convert the text data into a format
    # suitable for the model
    vectorizer = CountVectorizer(stop_words=stopwords.words("english"))

    # Creating a Naive Bayes classifier
    clf = MultinomialNB()

    # Combining the vectorizer and classifier into a single pipeline
    return make_pipeline(vectorizer, clf)

class Distinguish_filter(Filter):
    """Filter that removes samples that are too different from real data."""

    model: Pipeline
    text_column: str
    real_texts: List[str]
    synthetic_texts: List[str]
    real_data_amount: int
    collecting_data: bool
    score_threshold: float

    def configure(self, config: Dict):
        """
        Configure the filter

        :param config: The config dictionary
        """

        self.model = make_NB_model()
        self.text_column = config["text_column"]
        self.real_texts = pd.read_csv(config["real_data_location"])[self.text_column].tolist()
        self.real_data_amount = len(self.real_texts)

        if ("synthetic_data_location" in config):
            self.synthetic_texts = pd.read_csv(config["synthetic_data_location"])[self.text_column].tolist()[:self.real_data_amount]
            self._fit_model_beginning()
            self.collecting_data = False
        else:
            self.collecting_data = True

        self.score_threshold = config["score_threshold"]
        

    def filter(self, samples: List[Dict], dataset_handler: Dataset_handler) -> List[Dict]:

        if (self.collecting_data):
            if (len(dataset_handler.dataset) == self.real_data_amount):
                self._fit_model(dataset_handler)
                self.collecting_data = False
            else:
                return samples

        results = []
        # Filter all labels in the list
        for sample_dict in samples:
            sample = sample_dict["sample"]
            prob_i, _ = self.model.predict_proba([sample])[0]

            if(prob_i < self.score_threshold):
                results.append(sample_dict)

        return results

    def _fit_model_beginning(self) -> None:
        real_labels = [1] * len(self.real_texts)
        generated_labels = [0] * len(self.synthetic_texts)

        texts = self.real_texts + self.synthetic_texts
        labels = real_labels + generated_labels

        self.model.fit(texts, labels)

    def _fit_model(self, dataset_handler: Dataset_handler):
        real_texts = self.real_texts
        real_labels= [1] * len(real_texts)

        generated_texts = dataset_handler.dataset[self.text_column].tolist()
        generated_labels = [0] * len(generated_texts)

        texts = real_texts + generated_texts
        labels = real_labels + generated_labels

        self.model.fit(texts, labels)

        dataset_handler._initialize_empty_dataset()

    @staticmethod
    def get_name() -> str:
        """
        Get the name of the dataset handler

        :return: The name of the dataset handler
        """
        return "distinguishing_filter"
