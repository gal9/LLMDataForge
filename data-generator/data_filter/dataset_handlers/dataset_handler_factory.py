from .dataset_handler import Dataset_handler


class Dataset_handler_factory():
    @staticmethod
    def create_and_configure(config: dict) -> Dataset_handler:
        """Finds and returns the correct configured Dataset_handler class.

        :param config: configuration dictionary
        :raises ValueError: if the class name is invalid
        :return: The configured Dataset_handler class
        """
        # Get the adapter name
        try:
            class_name = config["dataset_handler_configuration"]["name"]
        except KeyError:
            raise ValueError("Incorrect configuration. No class name provided")

        # Get all classes
        available_classes = {cls.get_name(): cls for cls in Dataset_handler.__subclasses__()}

        # Create and configure the adapter
        if class_name in available_classes:
            dataset_handler = available_classes[class_name]()
            dataset_handler.configure(config)
            return dataset_handler
        else:
            raise ValueError("Invalid class name")
