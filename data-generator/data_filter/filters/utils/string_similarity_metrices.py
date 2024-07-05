from abc import ABC, abstractmethod
from typing import Dict


class String_similarity_metric(ABC):
    @abstractmethod
    def configure(self, config: Dict) -> None:
        pass

    @abstractmethod
    def similarity(self, text1: str, text2: str) -> float:
        pass

    @abstractmethod
    def get_name() -> str:
        pass


class String_similarity_metric_factory():
    @staticmethod
    def create_and_configure(config: dict) -> String_similarity_metric:
        """Finds and returns the correct configured filter class.

        :param config: configuration dictionary
        :raises ValueError: if the class name is invalid
        :return: The configured Filter class
        """
        # Get the adapter name
        try:
            class_name = config.get("name")
        except KeyError:
            raise ValueError("Incorrect configuration. No class name provided")

        # Get all classes
        available_classes = {cls.get_name(): cls for cls in String_similarity_metric.__subclasses__()}

        # Create and configure the adapter
        if class_name in available_classes:
            filter = available_classes[class_name]()
            filter.configure(config)
            return filter
        else:
            raise ValueError("Invalid class name")


class Equality(String_similarity_metric):
    def configure(self, config: Dict) -> None:
        return super().configure(config)

    def similarity(self, text1: str, text2: str) -> float:
        return 1 if text1 == text2 else 0

    @staticmethod
    def get_name() -> str:
        return "equality"
