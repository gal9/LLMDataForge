from .filter import Filter
from data_filter.dataset_handlers import Dataset_handler

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import random
from typing import Dict, List
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, test_set_location: str, dataset_handler: Dataset_handler, text_column: str = "text", label_column: str = "label") -> float:
        pass


class Classification_evaluator(Evaluator):
    @staticmethod
    def make_NB_model() -> Pipeline:
        # Creating a Count Vectorizer to convert the text data into a format
        # suitable for the model
        vectorizer = CountVectorizer(stop_words=stopwords.words("english"))

        # Creating a Naive Bayes classifier
        clf = MultinomialNB()

        # Combining the vectorizer and classifier into a single pipeline
        return make_pipeline(vectorizer, clf)

    def evaluate(self, test_set_location: str, dataset_handler: Dataset_handler, text_column: str = "text", label_column: str = "label") -> float:
        np.random.seed(1)
        random.seed(1)

        # Read test set as pandas dataframe
        test_set = pd.read_csv(test_set_location)

        generated_df = dataset_handler.dataset

        # Splitting the dataset into features and target variable
        X_test = test_set[text_column]
        y_test = test_set[label_column]

        X_train = generated_df[text_column]
        y_train = generated_df[label_column]

        model = Classification_evaluator.make_NB_model()

        # Train the model
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # Cast all predictions and y_test to lowercase
        predictions = np.char.lower(np.array(predictions).astype(str))
        y_test = np.char.lower(np.array(y_test).astype(str))

        # Return accuracy 4 decimal places
        return accuracy_score(y_test, predictions)


class Downstream_task_filter(Filter):
    """Evaluates the generated dataset against a downstream task every n samples. If the result gets worse throw the batch away."""

    batch_size: int
    score_threshold: float
    evaluator: Evaluator
    reference_data_location: str
    text_column: str
    label_column: str

    last_evaluated_at: int
    last_score: float

    def configure(self, config: Dict):
        """
        Configure the dataset handler

        :param config: The config dictionary
        """
        super().configure(config)

        self.batch_size = config["batch_size"]
        self.score_threshold = config["score_threshold"]
        self.reference_data_location = config["reference_data_location"]

        task_configuration = config["downstream_task"]
        task_name = task_configuration["name"]
        if (task_name == "classification"):
            self.evaluator = Classification_evaluator()
        else:
            raise ValueError(f"Unknown task {task_name}")

        self.text_column = config.get("text_column", "text")
        self.label_column = config.get("label_column", "label")

        self.last_evaluated_at = 0
        self.last_score = 0

    def filter(self, samples: List[Dict], dataset_handler: Dataset_handler) -> List[Dict]:
        number_of_samples = dataset_handler.get_dataset_size()

        if (number_of_samples % self.batch_size == 0 and number_of_samples != self.last_evaluated_at):
            self.logger.info("Evaluating dataset")
            # Evaluate the dataset
            current_score = self.evaluator.evaluate(
                self.reference_data_location,
                dataset_handler,
                text_column=self.text_column,
                label_column=self.label_column)

            # If the score is worse, remove the last batch
            if (self.last_score - current_score > self.score_threshold):
                self.logger.info(f"Current score: {current_score}, last score: {self.last_score}")
                # Now wtire this to file also
                with open("failed_downstream.txt", "a") as f:
                    f.write(f"Current score: {current_score}, last score: {self.last_score}\n")
                # remove last batch of samples
                dataset_handler.remove_last_n_samples(self.batch_size)
                print("removing")
            else:
                self.last_score = current_score
                self.last_evaluated_at = number_of_samples
                print("passed")

        return samples

    @staticmethod
    def get_name() -> str:
        """
        Get the name of the dataset handler

        :return: The name of the dataset handler
        """
        return "filter_downstream_task"
