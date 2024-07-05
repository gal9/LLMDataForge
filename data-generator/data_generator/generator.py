import logging
from typing import Dict

from .sample_generators import Sample_generator, Sample_generator_factory
from data_filter import Data_filter


# Load logger configuration from a file
try:
    logging.config.fileConfig("logger_config.ini")
except Exception:
    logging.config.fileConfig("data-generator/logger_config.ini")
logger = logging.getLogger(__name__)


class Data_generator():
    target_sample_number: int
    sample_generator: Sample_generator
    data_filter: Data_filter

    def configure(self, config: Dict) -> None:
        # Initialize and configure sample generator
        self.sample_generator = Sample_generator_factory().create_and_configure(config)
        logger.info("Sample generator configuration finished")

        # Initialize and configure data filter
        self.data_filter = Data_filter()
        self.data_filter.configure(config)
        logger.info("Data filter configuration finished")

        # Data generator configuration
        self.target_sample_number = config["target_sample_number"]
        logger.info("Data generator configuration finished")

    def generate_dataset(self) -> None:
        # Get initial dataset summary
        dataset_summary = self.data_filter.dataset_handler.summarize_dataset()

        # Generate samples until the target number is reached
        while (dataset_summary.number_of_samples < self.target_sample_number):
            # Log number of samples generated
            logger.info(f"Number of samples generated: {dataset_summary.number_of_samples}")

            # Generate a sample
            sample = self.sample_generator.generate_sample(dataset_summary)

            # Process sample with the filter
            self.data_filter.filter_data(sample)

            dataset_summary = self.data_filter.dataset_handler.summarize_dataset()

            print(dataset_summary.number_of_samples, flush=True)
