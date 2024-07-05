from typing import Any, List

import numpy as np

from .sample_generator import Sample_generator
from data_filter.dataset_handlers.classification_dataset_handler import Custom_dataset_summary
from data_filter.dataset_handlers import Dataset_summary


class Classification_sample_generator(Sample_generator):
    labels: List[str]
    target_distribution: List[int]
    length_mean: float
    length_std: float

    def __init__(self) -> None:
        super().__init__()

    def configure(self, config: dict) -> None:
        super().configure(config)

        component_configuration = config["sample_generator_configuration"]

        self.labels = component_configuration["labels"]

        # Read sample length parameters
        self.length_mean = component_configuration.get("length_mean", None)
        self.length_std = component_configuration.get("length_std", None)

        # Set target distribution
        target_distribution = component_configuration.get("target_distribution", "uniform")
        if (target_distribution == "uniform"):
            self.target_distribution = [1] * len(self.labels)
        else:
            self.target_distribution = target_distribution

        # verify that the labels and target distribution are of the same length
        assert len(self.labels) == len(self.target_distribution), "Labels and target distribution have to be of the same length"

    def generate_sample(self, dataset_summary: Dataset_summary) -> Any:
        values_to_insert_in_prompt = {}

        target_label = self._get_required_label(dataset_summary)
        values_to_insert_in_prompt["target_label"] = target_label

        if (self.length_mean is not None and self.length_std is not None):
            values_to_insert_in_prompt["sample_length"] = self._get_sample_length()

        prompt = self.prompt

        # Adding additional data to prompt
        additional_data_generator_metadata = {"label": target_label}
        for additional_data_generator in self.additional_data_generators:
            field = additional_data_generator.prompt_placeholder

            # Get reference samples
            additional_text = additional_data_generator.get_sample(additional_data_generator_metadata)

            values_to_insert_in_prompt[field] = additional_text

        prompt = prompt.format(**values_to_insert_in_prompt)

        # Call LLM
        response = self.llm_adapter.call_llm(prompt)

        # Construct the dictionary to return
        result_dict = {
            "sample": response,
            "label": target_label
        }

        return [result_dict]

    def _get_required_label(self, dataset_summary: Custom_dataset_summary) -> str:
        """Return the label for which the most samples are currently required based on the dataset statistics and the target distribution.

        :param dataset_statistics: A dictionary containing the statistics for the dataset including fields 'count_by_class' and
        'number_of_samples'.
        :return: The label for which the most samples are currently required.
        """

        number_of_samples = dataset_summary.number_of_samples
        number_of_samples_per_label = dataset_summary.number_of_samples_per_label

        # Set labels that do not exist to 0
        for label in self.labels:
            if (label not in number_of_samples_per_label):
                number_of_samples_per_label[label] = 0

        # Based on the target distribution determine which label is currently required
        target_distribution_sum = sum(self.target_distribution)
        most_missing = 0
        label_to_return = self.labels[0]
        for i, label in enumerate(self.labels):
            # Get count and the count that should be reached for this label and compute how many are missing
            count = number_of_samples_per_label[label]
            target_count = number_of_samples * self.target_distribution[i] / target_distribution_sum
            missing = target_count - count

            # Check if the most missing count is greater than the current count
            if (missing > most_missing):
                most_missing = missing
                label_to_return = label

        return label_to_return

    # def _generate_sample_without_reference(self, dataset_summary: Custom_dataset_summary) -> List[Dict]:
    #     target_label = self._get_required_label(dataset_summary)

    #     prompt = self.without_reference_prompt

    #     if (self.length_mean is not None):
    #         # Get sample length
    #         sample_length = self._get_sample_length()

    #         # Add target label
    #         prompt = prompt.format(target_label=target_label, sample_length=sample_length)
    #     else:
    #         prompt = prompt.format(target_label=target_label)

    #     # Call LLM
    #     response = self.llm_adapter.call_llm(prompt)

    #     # Construct the dictionary to return
    #     result_dict = {
    #         "sample": response,
    #         "label": target_label
    #     }

    #     return [result_dict]

    # def _generate_sample_with_reference(self, dataset_statistics: Dict) -> List[Dict]:
    #     target_label = self._get_required_label(dataset_statistics)

    #     # Adding additional data to prompt
    #     additional_data_generator_metadata = {"label": target_label}
    #     values_to_insert_in_prompt = {}
    #     for additional_data_generator in self.additional_data_generators:
    #         field = additional_data_generator.prompt_placeholder

    #         # Get reference samples
    #         additional_text = additional_data_generator.get_sample(additional_data_generator_metadata)

    #         prompt = self.with_reference_prompt

    #         if (self.length_mean is not None):
    #             # Get sample length
    #             sample_length = self._get_sample_length()

    #             prompt = prompt.format(target_label=target_label, reference_texts=reference_texts, sample_length=sample_length)
    #         else:
    #             prompt = prompt.format(target_label=target_label, reference_texts=reference_texts)

    #     # Call LLM
    #     response = self.llm_adapter.call_llm(prompt)

    #     # Construct the dictionary to return
    #     result_dict = {
    #         "sample": response,
    #         "label": target_label
    #     }

    #     return [result_dict]

    def _get_sample_length(self) -> int:
        """
        Get the length of the sample based on a normal distribution

        :return: The length of the sample
        """

        return np.round(np.random.normal(self.length_mean, self.length_std)).astype(int)

    @staticmethod
    def get_name() -> str:
        return "classification"
