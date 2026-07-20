import os


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")

    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in {"'", '"'}
    ):
        return value[1:-1]

    return value


class AzureAISearchConfig:
    def __init__(self) -> None:
        self.endpoint = required_env("AZURE_AI_SEARCH_ENDPOINT")
        self.api_key = required_env("AZURE_AI_SEARCH_API_KEY")
        self.index_name = required_env("AZURE_AI_SEARCH_INDEX_NAME")
        self.api_version = required_env("AZURE_AI_SEARCH_API_VERSION")


def load_config() -> AzureAISearchConfig:
    return AzureAISearchConfig()
