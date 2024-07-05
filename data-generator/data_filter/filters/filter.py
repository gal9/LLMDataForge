from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List

from data_filter.dataset_handlers import Dataset_handler


class Filter(ABC):
    logger: logging.Logger
    dataset_file_name_initialize: str
    dataset: Any

    @abstractmethod
    def configure(self, config: Dict):
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

    @abstractmethod
    def filter(self, samples: List[Dict], dataset_handler: Dataset_handler) -> List[Dict]:
        pass

    @abstractmethod
    def get_name() -> str:
        """
        Get the name of the dataset handler

        :return: The name of the dataset handler
        """
        pass
