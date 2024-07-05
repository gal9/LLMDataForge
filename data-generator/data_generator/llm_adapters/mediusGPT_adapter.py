import requests
from .LLM_adapter import LLM_adapter


class MediusGPT_adapter(LLM_adapter):
    model_url: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure(self, config: dict):
        component_configuration = config["LLM_adapter_configuration"]

        self.model_url = component_configuration["model_url"]
        self.certificate_location = component_configuration.get("certificate_location", False)

        return super().configure(config)

    def call_llm(self, query: str) -> str:
        """Functions sends a get request to the model_url and returns the answer.

        :param query: The query to be sent to the model.
        :return: The answer of the model.
        """
        # Format the query
        request_url = f"{self.model_url}/ask?query={query}&use_private_knowledgebase=false"

        # Send the request
        try:
            r = requests.get(request_url, headers={"Content-Type": "application/json"}, verify=self.certificate_location)
            # TODO add SSL verification

            # Check that the response is ok
            assert r.status_code == 200

            # Get the answer
            return r.json()["answer"]
        # Handle status code exception
        except AssertionError:
            raise Exception(f"Failed to call model: with status code {r.status_code}.")
        # Handle request exception
        except requests.exceptions.RequestException:
            raise Exception("Failed to call model.")

    @staticmethod
    def get_name() -> str:
        return "MediusGPT"
