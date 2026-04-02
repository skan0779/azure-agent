import uvicorn

from azure_agent.api.app import create_app

app = create_app()


def main() -> None:
    uvicorn.run(
        "azure_agent.api.main:app",
        host="0.0.0.0",
        port=8080,
    )


if __name__ == "__main__":
    main()
