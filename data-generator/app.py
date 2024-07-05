import argparse
import json
import logging
import logging.config
import os
import sys


current_dir = os.getcwd()
project_root = os.path.dirname(current_dir)
# Add the 'src' directory to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from data_generator import Data_generator # noqa

# Load logger configuration from a file
try:
    logging.config.fileConfig("logger_config.ini")
except Exception:
    logging.config.fileConfig("data-generator/logger_config.ini")
logger = logging.getLogger(__name__)


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process a JSON configuration file.")

    # Add the --config_file argument
    parser.add_argument("--config_file", type=str, help="JSON configuration file path")

    # Parse the arguments
    args = parser.parse_args()

    # Read and process the JSON configuration file
    if args.config_file:
        try:
            with open(args.config_file, "r") as file:
                config_data = json.load(file)
                logger.info(f"Configuration data from {args.config_file}:")
                logger.info(json.dumps(config_data, indent=4))
        except Exception as e:
            logger.error(f"Error reading configuration file: {e}")
    else:
        logger.warning("No configuration file provided.")

    # Create the DataGenerator object
    data_generator = Data_generator()

    # Configure the data generator
    data_generator.configure(config_data)

    # Rune the data generation
    logger.info("Starting the data generation.")
    data_generator.generate_dataset()


if __name__ == "__main__":
    main()
