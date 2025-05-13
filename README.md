# product_search_rag_YOUR_NUMBER-python-srv

This is the Python-based service for the Product Search RAG application. It integrates with SAP HANA Cloud and Generative AI models to provide retrieval-augmented generation (RAG) capabilities.

## Features

- Flask-based API for processing user queries and interacting with AI models.
- SAP HANA integration for semantic retrieval using vector search.
- Generative AI Hub SDK for embedding generation and chat completions.
- Secure authentication using SAP XSUAA.
- RAG workflow combining vector search results with AI-generated responses.

## Prerequisites

- Python 3.11.x
- SAP HANA Cloud instance
- Cloud Foundry CLI
- Required Python dependencies (see `requirements.txt`)

## Environment Configuration

1. **Set Up Environment Variables**:
   - Update the `env_cloud.json` file with your SAP HANA Cloud credentials.
   - Update the `env_config.json` file with your AI Core configuration, including API keys and endpoints.

2. **Run Locally**:
Use the following command to start the Flask server locally:

``` bash
python3 server.py
```

The service will be accessible at `http://localhost:3000`.

## Deployment

1. **Deploy to Cloud Foundry**:
Run the deployment script:

``` bash
bash deploy.sh
```

2. **Manifest Configuration**:
Ensure the `manifest.yml` file is correctly configured with your application name and bound services.

## Endpoints

### `/retrieveData` (POST)
Processes user queries and returns AI-generated responses.

#### Request Body:
The request body should be in the following format:
``` json
{
    "query": "Your question here",
    "prompt": "System prompt",
    "chatModelName": "gpt-4o",
    "topK": 15,
    "withRAG": true
}
```

#### Response:
The response will be in the following format:
``` json
{
    "result": "AI-generated response",
    "error": "Error message (if any)"
}
```

## Key Files

- `server.py`: Main Flask application.
- `server_structured_response.py`: Handles structured responses using Pydantic models.
- `deploy.sh`: Deployment script for Cloud Foundry.
- `manifest.yml`: Cloud Foundry application manifest.
- `requirements.txt`: Python dependencies.

## Security

- Authentication is managed via SAP XSUAA.
- Ensure sensitive credentials are stored securely in `env_cloud.json` and `env_config.json`.

## License

This project is licensed under the Apache 2.0 License.