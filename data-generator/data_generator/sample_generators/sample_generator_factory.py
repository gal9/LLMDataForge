from typing import Dict
from .sample_generator import Sample_generator


class Sample_generator_factory():
    @staticmethod
    def create_and_configure(config: Dict) -> Sample_generator:
        """Finds and returns the correct configured sample adapter class.

        :param config: configuration dictionary
        :raises ValueError: if the class name is invalid
        :return: The configured sample generator class
        """
        # Get the adapter name
        try:
            class_name = config["sample_generator_configuration"]["name"]
        except KeyError:
            raise ValueError("Incorrect configuration. No class name provided")

        # Get all classes
        available_classes = {cls.get_name(): cls for cls in Sample_generator.__subclasses__()}

        # Create and configure the sample generator class
        if class_name in available_classes:
            sample_generator = available_classes[class_name]()
            sample_generator.configure(config)
            return sample_generator
        else:
            raise ValueError("Invalid class name")
