from typing import Dict
import requests
from .LLM_adapter import LLM_adapter


class Mistral_adapter(LLM_adapter):
    model_url: str
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    system_prompt: str
    base_prompt: str = """[INST]
{system_prompt}

{user_message}
[/INST]
"""
    request_body: Dict = {
        "inputs": None,
        "parameters": {
            "best_of": 1,
            "decoder_input_details": True,
            "details": True,
            "do_sample": True,
            "max_new_tokens": 1024,
            "repetition_penalty": 1.03,
            "return_full_text": False,
            "seed": None,
            "stop": [
                "photographer"
            ],
            "temperature": None,
            "top_k": None,
            "top_n_tokens": 0,
            "top_p": None,
            "truncate": None,
            "typical_p": 0.95,
            "watermark": True
        }
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure(self, config: dict):
        component_configuration = config["LLM_adapter_configuration"]

        # Connection configuration
        self.model_url = Mistral_adapter._get_config(component_configuration, "model_url")
        self.certificate_location = Mistral_adapter._get_config(component_configuration, "certificate_location", False)

        # LLM call configuration (also save it to request body so that it is ready)
        self.temperature = Mistral_adapter._get_config(component_configuration, "temperature", 0.5)
        self.request_body["parameters"]["temperature"] = self.temperature
        self.top_k = Mistral_adapter._get_config(component_configuration, "top_k", 10)
        self.request_body["parameters"]["top_k"] = self.top_k
        self.top_p = Mistral_adapter._get_config(component_configuration, "top_p", 0.95)
        self.request_body["parameters"]["top_p"] = self.top_p
        self.repetition_penalty = Mistral_adapter._get_config(component_configuration, "repetition_penalty", 1.03)
        self.request_body["parameters"]["repetition_penalty"] = self.repetition_penalty

        # Prompt configuration
        self.system_prompt = Mistral_adapter._get_config(component_configuration, "system_prompt", "You are a helpful assistant.")

        return super().configure(config)

    def call_llm(self, query: str) -> str:
        """Functions sends a get request to the model_url and returns the answer.

        :param query: The query to be sent to the model.
        :return: The answer of the model.
        """

        # Prepare prompt
        prompt = self.base_prompt.format(system_prompt=self.system_prompt, user_message=query)
        self.request_body["inputs"] = prompt

        # Send the request
        try:
            r = requests.post(self.model_url, json=self.request_body, headers={"Content-Type": "application/json"},
                              verify=self.certificate_location)
            # TODO add SSL verification

            # Check that the response is ok
            assert r.status_code == 200

            # Get the answer
            return r.json()["generated_text"]
        # Handle status code exception
        except AssertionError:
            self.logger.warning(f"Failed to call model: with status code {r.status_code}.")
        # Handle request exception
        except requests.exceptions.RequestException:
            self.logger.warning("Failed to call model.")

    @staticmethod
    def get_name() -> str:
        return "mistral"
