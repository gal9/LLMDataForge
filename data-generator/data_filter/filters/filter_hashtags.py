import re
from typing import Dict, List
from .filter import Filter
from data_filter.dataset_handlers import Dataset_handler


class Hashtag_filter(Filter):
    """Filter that removes hashtags from the list of samples."""

    # removes any word starting with #
    hashtag_pattern: re.Pattern = re.compile(r"#\w+")

    def configure(self, config: Dict):
        """
        Configure the filter

        :param config: The config dictionary
        """
        pass

    def filter(self, samples: List[Dict], dataset_handler: Dataset_handler) -> List[Dict]:
        # Filter all characters in the list
        for sample_dict in samples:
            # Remove all characters from character_list from field "sample"
            sample_dict["sample"] = self.hashtag_pattern.sub(r"", sample_dict["sample"])

        return samples

    @staticmethod
    def get_name() -> str:
        """
        Get the name of the dataset handler

        :return: The name of the dataset handler
        """
        return "remove_hashtags"
