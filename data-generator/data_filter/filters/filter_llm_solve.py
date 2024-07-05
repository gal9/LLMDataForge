from typing import Any, Dict, List
from .filter import Filter
from data_filter.dataset_handlers import Dataset_handler
from data_generator.llm_adapters import LLM_adapter, LLM_adapter_factory


class LLM_solve_filter(Filter):
    """That calls an arbitrary LLM which preforms the downstream task and if the result does not match the label the sample is rejected."""

    llm: LLM_adapter
    base_prompt: str

    def configure(self, config: Dict):
        """
        Configure the filter

        :param config: The config dictionary
        """

        llm_adapter_configuration = config["LLM_adapter_configuration"]
        # Fill LLM parameters
        if ("temperature" not in llm_adapter_configuration):
            llm_adapter_configuration["temperature"] = 0.01
        if ("top_k" not in llm_adapter_configuration):
            llm_adapter_configuration["top_k"] = 20
        if ("top_p" not in llm_adapter_configuration):
            llm_adapter_configuration["top_p"] = 0.90
        if ("repetition_penalty" not in llm_adapter_configuration):
            llm_adapter_configuration["repetition_penalty"] = 1.03

        self.llm = LLM_adapter_factory.create_and_configure({"LLM_adapter_configuration": llm_adapter_configuration})

        self.base_prompt = config["prompt"]

    def filter(self, samples: List[Dict], dataset_handler: Dataset_handler) -> List[Dict]:
        results = []

        for sample_dict in samples:
            sample = sample_dict["sample"]
            label = sample_dict["label"]

            prompt = self.base_prompt.format(sample=sample)

            res = self.llm.call_llm(prompt)

            if (self.check_response(res, label)):
                results.append(sample_dict)
            else:
                # Add sample to txt file
                with open("failed.txt", "a") as f:
                    f.write(sample + res + label + "\n")

        return results

    def check_response(self, response: str, target: Any) -> bool:
        prediction = response.splitlines()[0]

        # Case insensitive check if target is contained in prediction
        if (target.lower() in prediction.lower()):
            return True

        return False

    @staticmethod
    def get_name() -> str:
        """
        Get the name of the dataset handler

        :return: The name of the dataset handler
        """
        return "llm_solve_filter"
