from .LLM_adapter import LLM_adapter


class LLM_adapter_factory():
    @staticmethod
    def create_and_configure(config: dict) -> LLM_adapter:
        """Finds and returns the correct configured LLM_adapter class.

        :param config: configuration dictionary
        :raises ValueError: if the class name is invalid
        :return: The configured LLM_adapter class
        """
        # Get the adapter name
        try:
            class_name = config["LLM_adapter_configuration"]["name"]
        except KeyError:
            raise ValueError("Incorrect configuration. No class name provided")

        # Get all classes
        available_classes = {cls.get_name(): cls for cls in LLM_adapter.__subclasses__()}

        # Create and configure the adapter
        if class_name in available_classes:
            adapter = available_classes[class_name](class_name)
            adapter.configure(config)
            return adapter
        else:
            raise ValueError("Invalid class name")
