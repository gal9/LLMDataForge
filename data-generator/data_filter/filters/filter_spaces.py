from typing import Dict, List
from .filter import Filter
from data_filter.dataset_handlers import Dataset_handler


class Spaces_filter(Filter):
    """Filter that removes spaces from the samples."""

    def configure(self, config: Dict):
        """
        Configure the filter

        :param config: The config dictionary
        """
        super().configure(config)

    def filter(self, samples: List[Dict], dataset_handler: Dataset_handler) -> List[Dict]:
        # Strip samples from the samples
        for sample_dict in samples:
            sample_dict["sample"] = sample_dict["sample"].strip()

        return samples

    @staticmethod
    def get_name() -> str:
        """
        Get the name of the dataset handler

        :return: The name of the dataset handler
        """
        return "remove_spaces"
