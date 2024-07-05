from typing import Dict, List

import pandas as pd
from .dataset_handler import Dataset_handler
from .dataset_summary import Dataset_summary


class Custom_dataset_summary(Dataset_summary):
    number_of_samples_per_label: Dict[str, int]

    def __init__(self, number_of_samples, number_of_samples_per_label):
        self.number_of_samples_per_label = number_of_samples_per_label
        super().__init__(number_of_samples)


class Classification_dataset_handler(Dataset_handler):
    dataset: pd.DataFrame

    def configure(self, config: dict):
        component_configuration = config["dataset_handler_configuration"]

        # Read dataframe columns from configuration. The first element in the list is the column sample and the second is column label
        self.dataframe_columns = component_configuration.get("dataframe_columns", ["sample", "label"])

        return super().configure(config)

    def _initialize_empty_dataset(self) -> None:
        # Initialize empty pandas dataset with specified columns
        self.dataset = pd.DataFrame(columns=self.dataframe_columns)

    def _initialize_from_file(self, path: str) -> None:
        # Initialize pandas dataframe from file
        try:
            self.dataset = pd.read_csv(path)
        except FileNotFoundError:
            self.logger.warning(f"Dataset initialization file not found at {path}. Starting from scratch")
            self._initialize_empty_dataset()

    def get_list_of_samples(self) -> List:
        # Return the list of samples from the dataframe column sample
        return self.dataset[self.dataframe_columns[0]].tolist()

    def save_to_file(self) -> None:
        # Save the dataframe to a csv file
        self.dataset.to_csv(self.dataset_save_file_name, index=False)

    def summarize_dataset(self) -> Custom_dataset_summary:
        # Get number of samples in the dataset
        number_of_samples = len(self.dataset)

        # get number of samples for every label
        labels = self.dataset[self.dataframe_columns[1]].unique()
        number_of_samples_per_label = {label: len(self.dataset[self.dataset[self.dataframe_columns[1]] == label]) for label in labels}

        summary = Custom_dataset_summary(number_of_samples, number_of_samples_per_label)

        return summary

    def add_to_dataset(self, samples: List[Dict]) -> None:
        # Iterate over the list of samples and insert the sample into the dataset
        for sample in samples:
            sample_str = sample["sample"]
            sample_label = sample["label"]

            # insert sample to dataset where the first element of the list is the column sample and the second is column label
            self.dataset.loc[len(self.dataset)] = [sample_str, sample_label]

    def get_dataset_size(self) -> int:
        return self.dataset.shape[0]

    def remove_last_n_samples(self, n: int) -> None:
        self.dataset = self.dataset[:-n]

    @staticmethod
    def get_name() -> str:
        return "classification"
