from typing import Dict, List
from .filter import Filter
from data_filter.dataset_handlers import Dataset_handler


class Custom_emotion_task_filter(Filter):
    """Custom filter for any specifics of the emotion task."""

    def configure(self, config: Dict):
        """
        Configure the filter

        :param config: The config dictionary
        """
        pass

    def filter(self, samples: List[Dict], dataset_handler: Dataset_handler) -> List[Dict]:

        # Filter all characters in the list
        for sample_dict in samples:
            pass

        return samples

    @staticmethod
    def get_name() -> str:
        """
        Get the name of the dataset handler

        :return: The name of the dataset handler
        """
        return "custom_emotion_task"
