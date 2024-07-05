from abc import ABC, abstractmethod
import logging
from typing import Dict


class Additional_data_generator(ABC):
    data_file: str
    batch_size: int
    prompt_placeholder: str

    def __init__(self) -> None:
        super().__init__()

    def configure(self, config: Dict) -> None:
        # Set batch size if in config else set to 1
        self.batch_size = config.get("batch_size", 1)

        # Load logger configuration from a file
        try:
            logging.config.fileConfig("logger_config.ini")
        except Exception:
            logging.config.fileConfig("data-generator/logger_config.ini")
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def get_sample(self, metadata: Dict) -> str:
        pass

    @abstractmethod
    def get_name() -> str:
        pass
