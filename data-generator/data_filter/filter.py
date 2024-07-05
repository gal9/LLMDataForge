from typing import Any, Dict, List

from .dataset_handlers import Dataset_handler, Dataset_handler_factory
from .filters import Filter, Filter_factory


class Data_filter():
    dataset_handler: Dataset_handler
    filters: List[Filter]
    save_interval: int

    def configure(self, config: Dict) -> None:
        component_configuration = config["data_filter_configuration"]

        self.dataset_handler = Dataset_handler_factory.create_and_configure(config)

        filter_configurations = component_configuration["optional_filters"]

        self.save_interval = component_configuration.get("save_interval", None)

        # Create and configure filters
        self.filters = []
        for filter_configuration in filter_configurations:
            self.filters.append(Filter_factory.create_and_configure(filter_configuration))

    def filter_data(self, sample: Any) -> None:
        # Apply specified filters to the sample
        for filter in self.filters:
            try:
                sample = filter.filter(sample, self.dataset_handler)
            except Exception:
                return

        self.dataset_handler.add_to_dataset(sample)

        if (self.save_interval is not None and self.dataset_handler.get_dataset_size() % self.save_interval == 0):
            self.dataset_handler.save_to_file()
