import json
import random
from typing import Dict, List

from ..additional_data_generator import Additional_data_generator


class Random_unlabeled_reference_example_generator(Additional_data_generator):
    sample_list: List[Dict]

    def __init__(self) -> None:
        super().__init__()

    def configure(self, config: dict) -> None:
        super().configure(config)

        self.data_file = config["data_file"]

        # Open a json file which contains a list of objects and save the list
        self.sample_list = json.load(open(self.data_file, "r"))

        # Set random seed if specified
        if ("seed" in config):
            random.seed(config["seed"])

        self.prompt_placeholder = config.get("prompt_placeholder", "reference_texts")

    def get_sample(self, metadata: Dict) -> str:
        """Get the batch of samples from the list

        :return: The next batch of samples (first element is the data and second is the label).
        """

        # Select batch_size number of random samples from self.sample_list
        samples = random.sample(self.sample_list, self.batch_size)

        # Return the data and label
        return "\n".join([sample["data"] for sample in samples])

    @staticmethod
    def get_name() -> str:
        return "random_unlabeled_reference_example"
