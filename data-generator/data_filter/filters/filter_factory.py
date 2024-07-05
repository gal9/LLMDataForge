from .filter import Filter


class Filter_factory():
    @staticmethod
    def create_and_configure(config: dict) -> Filter:
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
        available_classes = {cls.get_name(): cls for cls in Filter.__subclasses__()}

        # Create and configure the adapter
        if class_name in available_classes:
            filter = available_classes[class_name]()
            filter.configure(config)
            return filter
        else:
            raise ValueError(f"Invalid class name {class_name}")
