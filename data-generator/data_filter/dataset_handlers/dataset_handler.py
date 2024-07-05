from abc import ABC, abstractmethod
import logging
import os
from typing import Any, List

from .dataset_summary import Dataset_summary


class Dataset_handler(ABC):
    logger: logging.Logger
    dataset_save_file_name: str
    dataset: Any

    def configure(self, config: dict):
        """
        Configure the dataset handler

        :param config: The config dictionary
        """

        # Load logger configuration from a file
        try:
            logging.config.fileConfig("logger_config.ini")
        except Exception:
            logging.config.fileConfig("data-generator/logger_config.ini")
        self.logger = logging.getLogger(__name__)

        component_configuration = config["dataset_handler_configuration"]

        self.dataset_save_file_name = component_configuration["dataset_file"]

        # Initialize the dataset
        if ("dataset_file_name_initialize" in component_configuration):
            self._initialize_from_file(component_configuration["dataset_file_name_initialize"])
        else:
            self._initialize_empty_dataset()

    def __del__(self):
        """On destroy the file should be saved (ensure that the directory exists then save the file)"""
        self._ensure_path()
        self.save_to_file()

    def _ensure_path(self) -> None:
        """Method checks if dataset_save_file_name location exists and if not it creates it.
        """
        directory_name = "/".join(self.dataset_save_file_name.split("/")[:-1])

        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

    @abstractmethod
    def _initialize_empty_dataset(self) -> None:
        """Method initializes an empty dataset"""
        pass

    @abstractmethod
    def _initialize_from_file(self, path: str) -> None:
        """Method initializes the dataset from the file"""
        pass

    @abstractmethod
    def get_list_of_samples(self) -> List:
        """From the dataset extract the list of samples

        :return: List of samples
        """
        pass

    @abstractmethod
    def save_to_file(self) -> None:
        """
        Save the dataset to file

        :param query: The prompt for the LLM
        """
        pass

    @abstractmethod
    def summarize_dataset(self) -> Dataset_summary:
        """Returns the summarization of the dataset

        :return: Dictionary with dataset properties
        """
        pass

    @abstractmethod
    def add_to_dataset(self, samples: Any) -> None:
        """Adds sample to the dataset

        :param sample: Sample to add to the dataset
        """
        pass

    @abstractmethod
    def get_dataset_size(self) -> int:
        """
        Get the size of the dataset

        :return: The size of the dataset
        """
        pass

    @abstractmethod
    def remove_last_n_samples(self, n: int) -> None:
        """
        Remove the last n samples from the dataset

        :param n: The number of samples to remove
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the dataset handler

        :return: The name of the dataset handler
        """
        pass
