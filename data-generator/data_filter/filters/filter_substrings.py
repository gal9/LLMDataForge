from typing import Dict, List
from .filter import Filter
from data_filter.dataset_handlers import Dataset_handler


class Substring_filter(Filter):
    """Filter that removes labels from the samples. Sometimes the labels are added to the end of samples."""

    substrings_in_beginning: List[str]
    substrings_in_end: List[str]
    substrings_general: List[str]

    def configure(self, config: Dict):
        """
        Configure the filter

        :param config: The config dictionary
        """

        self.substrings_in_beginning = config.get("substrings_in_beginning", [])
        self.substrings_in_end = config.get("substrings_in_end", [])
        self.substrings_general = config.get("substrings_general", [])

    def filter(self, samples: List[Dict], dataset_handler: Dataset_handler) -> List[Dict]:
        # Filter all labels in the list
        for sample_dict in samples:
            sample = sample_dict["sample"]

            # Remove from sample any strings from the end that are in substrings_in_end
            for substring in self.substrings_in_end:
                if (sample.endswith(substring)):
                    sample = sample[:-len(substring)]

            # Remove from sample any strings from the beginning that are in substrings_in_beginning
            for substring in self.substrings_in_beginning:
                if (sample.startswith(substring)):
                    sample = sample[len(substring):]

            # Remove from sample any strings that are in substrings_general
            for substring in self.substrings_general:
                sample = sample.replace(substring, "")

            sample_dict["sample"] = sample

        return samples

    @staticmethod
    def get_name() -> str:
        """
        Get the name of the dataset handler

        :return: The name of the dataset handler
        """
        return "remove_substrings"
