import json
import re
from typing import Dict, List
from .filter import Filter
from data_filter.dataset_handlers import Dataset_handler


class Isolate_samples_filter(Filter):
    def configure(self, config: Dict):
        """
        Configure the dataset handler

        :param config: The config dictionary
        """
        super().configure(config)

    def filter(self, samples: List[Dict], dataset_handler: Dataset_handler) -> List[Dict]:
        json_pattern = r'\{([^}]+)\}'
        json_match = re.search(json_pattern, samples[0]["sample"])

        if (not json_match):
            raise ValueError(f"The sample {samples[0]['sample']} does not contain a valid JSON string")

        try:
            json_of_sample = json.loads(json_match.group(0))
        except json.decoder.JSONDecodeError:
            sample_str = samples[0]["sample"]
            self.logger.warning(f"The sample {sample_str} is not a valid JSON string. Skipping the sample.")
            raise ValueError(f"The sample {sample_str} is not a valid JSON string")

        results_dict = {
            "sample": json_of_sample["sample"],
            "label": samples[0]["label"]
        }

        return [results_dict]

    @staticmethod
    def get_name() -> str:
        """
        Get the name of the dataset handler

        :return: The name of the dataset handler
        """
        return "isolate_samples"
