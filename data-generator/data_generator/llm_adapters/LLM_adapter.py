from abc import ABC, abstractmethod
import logging
import os
from typing import Any, Dict


class LLM_adapter(ABC):
    name: str

    def __init__(self, *args, **kwargs):
        pass

    def configure(self, config: dict):
        """
        Configure the adapter

        :param config: The config dictionary
        """

        try:
            logging.config.fileConfig("logger_config.ini")
        except Exception:
            logging.config.fileConfig("data-generator/logger_config.ini")
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _get_config(config: Dict, key: str, default: Any = None) -> Any:
        if (key in config):
            value = config[key]
            if (isinstance(value, str) and value.startswith("$")):
                # Read the value from the environment variable
                value = os.getenv(value[1:])
            return value

        elif (default is not None):
            # Save the default value so that it can be logged
            config[key] = default
            return default

        else:
            raise KeyError(f"The configuration key {key} does not exist in configuration {config}")

    @abstractmethod
    def call_llm(self, query: str) -> str:
        """
        Call llm

        :param query: The prompt for the LLM
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the adapter

        :return: The name of the adapter
        """
        pass
