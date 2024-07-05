from typing import Dict, List
from .filter import Filter
from data_filter.dataset_handlers import Dataset_handler


class Multiple_lines_filter(Filter):
    """Filter that removes labels from the samples. Sometimes the labels are added to the end of samples."""

    labels: List[str]

    def configure(self, config: Dict):
        """
        Configure the filter

        :param config: The config dictionary
        """
        super().configure(config)

    def filter(self, samples: List[Dict], dataset_handler: Dataset_handler) -> List[Dict]:
        """This filter checks if the sample contains multiple lines. If it does we assume each line is a separate sample.

        :param samples: List of samples
        :param dataset_handler: The dataset handler object
        :return: The list of updated samples
        """
        new_samples = []
        # Filter new lines
        for sample_dict in samples:
            sample = sample_dict["sample"]

            # Split by lines and create separate samples
            lines = sample.split("\n")
            for line in lines:
                new_sample = sample_dict.copy()
                new_sample["sample"] = line
                new_samples.append(new_sample)

        return new_samples

    @staticmethod
    def get_name() -> str:
        """
        Get the name of the dataset handler

        :return: The name of the dataset handler
        """
        return "multiple_lines_filter"
