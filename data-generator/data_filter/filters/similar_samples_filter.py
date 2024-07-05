from typing import Dict, List

from .filter import Filter
from data_filter.dataset_handlers import Dataset_handler
from .utils.string_similarity_metrices import String_similarity_metric, String_similarity_metric_factory


class Similar_samples_filter(Filter):
    similarity_metric: String_similarity_metric
    similarity_threshold: float

    def configure(self, config: Dict):
        """
        Configure the dataset handler

        :param config: The config dictionary
        """

        self.similarity_metric = String_similarity_metric_factory.create_and_configure({"name": config.get("similarity_metric", "equality")})  # noqa: E501

        self.similarity_threshold = config.get("similarity_threshold", 0)

    def filter(self, samples: List[Dict], dataset_handler: Dataset_handler) -> List[Dict]:
        previous_samples = dataset_handler.get_list_of_samples()

        # Iterate over samples and remove samples that are similar to the previous samples
        final_samples = []
        for sample in samples:
            if (not any([self.similarity_metric.similarity(sample["sample"], previous_sample) > self.similarity_threshold for previous_sample in previous_samples])):  # noqa: E501
                final_samples.append(sample)
            else:
                self.logger.info(f"Removing sample {sample['sample']} as it is similar to the previous sample")

        return final_samples

    @staticmethod
    def get_name() -> str:
        """
        Get the name of the filter

        :return: The name of the filter
        """
        return "similar_samples"
