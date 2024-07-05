import re
from typing import Dict, List
from .filter import Filter
from data_filter.dataset_handlers import Dataset_handler


class Character_filter(Filter):
    """Filter removes characters (specified in the list) from the sample. Additionally we have a separate list of characters to remove from
    the end of the sample."""

    character_list: List[str]
    character_list_end_of_string: List[str]
    filter_emojis: bool

    emoji_pattern: re.Pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )

    def configure(self, config: Dict):
        """
        Configure the dataset handler

        :param config: The config dictionary
        """
        # Load list of characters to remove from the sample
        if ("character_list" not in config):
            self.character_list = re.compile("")
        else:
            self.character_list = re.compile("[{character_list}]".format(character_list="".join(config["character_list"])))

        # Load list of characters to remove from the end of the sample
        self.character_list_end_of_string = config.get("character_list_end_of_string", [])

        # Filter emojis if required
        self.filter_emojis = config.get("filter_emojis", False)

    def filter(self, samples: List[Dict], dataset_handler: Dataset_handler) -> List[Dict]:
        # Filter all characters in the list
        for sample_dict in samples:
            # Remove all characters from character_list from field "sample"
            sample_dict["sample"] = self.character_list.sub(r"", sample_dict["sample"])

        # Filter emojis if required
        if (self.filter_emojis):
            # Remove all emojis from sample
            for sample_dict in samples:
                sample_dict["sample"] = self.emoji_pattern.sub(r"", sample_dict["sample"])

        # Filter characters from the end of string
        for sample_dict in samples:
            sample = sample_dict["sample"]
            # If sample ends with any of the labels or capitalized version of the label, remove it
            for character in self.character_list_end_of_string:
                if (sample.endswith(character)):
                    sample = sample[:-len(character)]
            sample_dict["sample"] = sample

        return samples

    @staticmethod
    def get_name() -> str:
        """
        Get the name of the dataset handler

        :return: The name of the dataset handler
        """
        return "remove_characters"
