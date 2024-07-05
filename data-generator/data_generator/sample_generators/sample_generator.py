from abc import ABC, abstractmethod
import logging
from typing import Any

from data_generator.llm_adapters import LLM_adapter, LLM_adapter_factory
from data_generator.additional_data_generator import Additional_data_generator, Additional_data_generator_factory

from data_filter.dataset_handlers import Dataset_summary


class Sample_generator(ABC):
    llm_adapter: LLM_adapter
    additional_data_generators: list[Additional_data_generator]
    prompt: str

    def __init__(self) -> None:
        super().__init__()

    def configure(self, config: dict) -> None:
        component_configuration = config["sample_generator_configuration"]

        # Initialize the LLM adapter
        self.llm_adapter = LLM_adapter_factory.create_and_configure(config)

        # Initialize the additional data generators example generator
        self.additional_data_generators = []
        for additional_data_generator_configuration in config.get("additional_data_generators_configuration", []):
            additional_data_generator = Additional_data_generator_factory.create_and_configure(additional_data_generator_configuration)
            self.additional_data_generators.append(additional_data_generator)

        self.prompt = component_configuration.get("prompt", "")

        # Load logger configuration from a file
        try:
            logging.config.fileConfig("logger_config.ini")
        except Exception:
            logging.config.fileConfig("data-generator/logger_config.ini")
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def generate_sample(self, dataset_summary: Dataset_summary) -> Any:
        pass

    @abstractmethod
    def get_name() -> str:
        pass
