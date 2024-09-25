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
    distance_threshold_scale: float
    collect_sample: bool

    vector_dataset: str

    def configure(self, config: Dict):
        """
        Configure the filter

        :param config: The config dictionary
        """

        super().configure(config)

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
        dataset_location = config.get("reference_dataset_location", "")
        self.vector_dataset = []
        try:
            with open(dataset_location, "r") as f:
                sample_list = json.load(f)

                for sample in sample_list:
                    self.vector_dataset.append(self.embedding_model.get_embedding(sample["data"]))
            self.vector_dataset = np.array(self.vector_dataset)
        except FileNotFoundError:
            self.logger.info("No reference dataset found. Similarity filter will start with empty dataset")
            self.vector_dataset = np.empty((0, 768))

        self.distance_threshold_scale = config["distance_threshold_scale"]

        if (len(self.vector_dataset) != 0):
            # Find the pair in dataset that is most similar (skip checking same vectors)
            self._update_distance_threshold()
            self.collect_sample = False
        else:
            self.distance_threshold = 0
            self.collect_sample = True
        
    def _update_distance_threshold_mean(self):
        lowest_distance = np.mean([self.distance_metric.compute_distance(vector1, vector2) for i, vector1 in enumerate(self.vector_dataset) for vector2 in self.vector_dataset[i+1:]]) # noqa
        self.distance_threshold = lowest_distance * self.distance_threshold_scale

    def _update_distance_threshold(self):
        lowest_distance = min([self.distance_metric.compute_distance(vector1, vector2) for i, vector1 in enumerate(self.vector_dataset) for vector2 in self.vector_dataset[i+1:]]) # noqa
        self.distance_threshold = lowest_distance * self.distance_threshold_scale

    def filter(self, samples: List[Dict], dataset_handler: Dataset_handler) -> List[Dict]:
        results = []

        for sample_dict in samples:
            sample = sample_dict["sample"]

            sample_embedding = self.embedding_model.get_embedding(sample).numpy()

            if(self.collect_sample):
                if(len(self.vector_dataset) == 100):
                    self._update_distance_threshold()
                    self.collect_sample = False
                    self.logger.info("Finished calibrating the distance threshold.")
                else:
                    if (not self.check_if_similar(sample_embedding)):
                        results.append(sample_dict)

                        # Add the embedding to the vector dataset
                        self.vector_dataset = np.vstack((self.vector_dataset, sample_embedding.reshape(1, -1)))

            else:
                if (not self.check_if_similar(sample_embedding)):
                    results.append(sample_dict)

                    # Add the embedding to the vector dataset
                    self.vector_dataset = np.vstack((self.vector_dataset, sample_embedding.reshape(1, -1)))
                else:
                    pass
                    # Add sample to txt file
                    #with open("failed_embedding.txt", "a") as f:
                    #    f.write(sample + "\n")

        return results

    def check_if_similar(self, sample_embedding: np.ndarray) -> bool:
        if(len(self.vector_dataset) == 0):
            return False

        distances = np.apply_along_axis(self.distance_metric.compute_distance, 1, self.vector_dataset, sample_embedding)
        return np.any(distances <= self.distance_threshold)

    @staticmethod
    def get_name() -> str:
        """
        Get the name of the dataset handler

        :return: The name of the dataset handler
        """
        return "remove_similar_embeddings"
