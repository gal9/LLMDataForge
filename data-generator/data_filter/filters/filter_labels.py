from typing import Dict, List
from .filter import Filter
from data_filter.dataset_handlers import Dataset_handler


class Labels_filter(Filter):
    """Filter that removes labels from the samples. Sometimes the labels are added to the end of samples."""

    labels: List[str]

    def configure(self, config: Dict):
        """
        Configure the filter

        :param config: The config dictionary
        """

        self.labels = config["labels"]

        # Add versions of the label
        versions = []
        for label in self.labels:
            versions.append(": " + label.capitalize())
            versions.append(":" + label.capitalize())
            versions.append(": " + label)
            versions.append(":" + label)

        self.labels = versions

    def filter(self, samples: List[Dict], dataset_handler: Dataset_handler) -> List[Dict]:
        # Filter all labels in the list
        for sample_dict in samples:
            sample = sample_dict["sample"]
            # If sample ends with any of the labels or capitalized version of the label, remove it
            for label in self.labels:
                if (sample.endswith(label)):
                    sample = sample[:-len(label)]
                    break
            sample_dict["sample"] = sample

        return samples

    @staticmethod
    def get_name() -> str:
        """
        Get the name of the dataset handler

        :return: The name of the dataset handler
        """
        return "remove_labels"
