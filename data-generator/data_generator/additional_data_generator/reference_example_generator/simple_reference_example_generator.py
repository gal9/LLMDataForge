import json
from typing import Dict, List

from ..additional_data_generator import Additional_data_generator


class Simple_reference_example_generator(Additional_data_generator):
    sample_list: List[Dict]
    sample_index: int

    def __init__(self) -> None:
        super().__init__()

    def configure(self, config: dict) -> None:
        super().configure(config)

        # Open a json file which contains a list of objects and save the list
        self.sample_list = json.load(open(self.data_file, "r"))

        # Set the current index to the first sample
        self.sample_index = 0

    def get_sample(self, metadata: Dict) -> List[List[str]]:
        """Get the batch of samples from the list

        :return: The next batch of samples (first element is the data and second is the label).
        """

        # Get the next sample from the list or return the first one if we ran out of samples
        samples = []
        index = self.sample_index
        for _ in range(self.batch_size):
            try:
                samples.append(self.sample_list[index])
                index += 1
            except IndexError:
                samples.append(self.sample_list[0])
                index = 0

        self.sample_index += 1

        # Return the data and label
        return [[sample["data"], sample["label"]] for sample in samples]

    @staticmethod
    def get_name() -> str:
        return "simple"
