import json
import random
from typing import Dict, List

from ..additional_data_generator import Additional_data_generator


class Classification_random_reference_example_generator(Additional_data_generator):
    sample_list: List[Dict]

    def __init__(self) -> None:
        super().__init__()

    def configure(self, config: dict) -> None:
        super().configure(config)

        configuration: Dict = config["reference_example_generator_configuration"]

        # Open a json file which contains a list of objects and save the list
        self.sample_list = json.load(open(self.data_file, "r"))

        # Set random seed if specified
        if ("seed" in configuration):
            random.seed(configuration["seed"])

    def get_sample(self, metadata: Dict) -> List[List[str]]:
        """Get the batch of samples from the list

        :return: The next batch of samples (first element is the data and second is the label).
        """

        if ("label" not in metadata):
            self.logger.warning("No label found in the metadata.")
            samples_with_label = self.sample_list
        else:
            label = metadata["label"]
            samples_with_label = [sample for sample in self.sample_list if sample["label"] == label]

        # Select batch_size number of random samples from self.sample_list
        samples = random.sample(samples_with_label, self.batch_size)

        # Return the data and label
        return [[sample["data"], sample["label"]] for sample in samples]

    @staticmethod
    def get_name() -> str:
        return "classification_random"
