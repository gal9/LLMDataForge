from .additional_data_generator import Additional_data_generator


class Additional_data_generator_factory():
    @staticmethod
    def create_and_configure(config: dict) -> Additional_data_generator:
        """Finds and returns the correct configured additional data generator class.

        :param config: configuration dictionary
        :raises ValueError: if the class name is invalid
        :return: The configured reference additional data generator class
        """
        # Get the generator name
        try:
            class_name = config["name"]
        except KeyError:
            raise ValueError("Incorrect configuration. No class name provided")

        # Get all classes
        available_classes = {cls.get_name(): cls for cls in Additional_data_generator.__subclasses__()}

        # Create and configure the sample generator class
        if class_name in available_classes:
            additional_data_generator = available_classes[class_name]()
            additional_data_generator.configure(config)
            return additional_data_generator
        else:
            raise ValueError("Invalid class name")
