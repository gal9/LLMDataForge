from abc import ABC, abstractmethod
import json
from typing import Dict, List
import numpy as np
from .filter import Filter
from data_filter.dataset_handlers import Dataset_handler

from src.embedding_models.bert import Bert_model # noqa
from src.embedding_models.embedding_model import Embedding_model # noqa


class Distance(ABC):
    @abstractmethod
    def compute_distance(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        pass


class Cosine_distance(Distance):
    def compute_distance(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        # Calculate cosine distance scaled to 0-1
        return max(0, 1 - (np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))))


class Similar_embeddings_filter(Filter):
    """Filter samples that have similar embeddings"""

    embedding_algorithm: str
    embedding_model: Embedding_model
    distance_metric: str
    distance_metric: Distance
    distance_threshold: float

    vector_dataset: str

    def configure(self, config: Dict):
        """
        Configure the filter

        :param config: The config dictionary
        """

        # Configure embedding algorithm
        self.embedding_algorithm = config["embedding_algorithm"]
        if (self.embedding_algorithm == "bert"):
            self.embedding_model = Bert_model()
        else:
            raise ValueError("Unknown embedding algorithm")

        if (config["distance_metric"] == "cosine"):
            self.distance_metric = Cosine_distance()
        else:
            raise ValueError("Unknown distance metric")

        # Compute embeddings for teh reference dataset
        # TODO: handle no reference dataset
        dataset_location = config.get("reference_dataset_location")
        self.vector_dataset = []
        with open(dataset_location, "r") as f:
            sample_list = json.load(f)

            for sample in sample_list:
                self.vector_dataset.append(self.embedding_model.get_embedding(sample["data"]))

        # Find the pair in dataset that is most similar (skip checking same vectors)
        lowest_distance = min([self.distance_metric.compute_distance(vector1, vector2) for i, vector1 in enumerate(self.vector_dataset) for vector2 in self.vector_dataset[i+1:]]) # noqa

        self.distance_threshold = lowest_distance * config["distance_threshold_scale"]

    def filter(self, samples: List[Dict], dataset_handler: Dataset_handler) -> List[Dict]:
        results = []

        for sample_dict in samples:
            sample = sample_dict["sample"]

            sample_embedding = self.embedding_model.get_embedding(sample)

            if (not self.check_if_similar(sample_embedding)):
                results.append(sample_dict)
            else:
                # Add sample to txt file
                with open("failed_embedding.txt", "a") as f:
                    f.write(sample + "\n")

        return results

    def check_if_similar(self, sample_embedding: np.ndarray) -> bool:
        for reference_embedding in self.vector_dataset:
            if (self.distance_metric.compute_distance(sample_embedding, reference_embedding) < self.distance_threshold):
                return True
        return False

    @staticmethod
    def get_name() -> str:
        """
        Get the name of the dataset handler

        :return: The name of the dataset handler
        """
        return "remove_similar_embeddings"
