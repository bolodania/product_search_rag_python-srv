
# product_search_rag_YOUR_NUMBER-python-srv

This is the Python-based service for the Product Search RAG application. It integrates with SAP HANA Cloud and Generative AI models to provide retrieval-augmented generation (RAG) capabilities.


## Features

- Flask-based API for processing user queries and interacting with AI models.
- SAP HANA integration for semantic retrieval using vector search.
- Generative AI Hub SDK for embedding generation and chat completions.
- Secure authentication using SAP XSUAA.
- RAG workflow combining vector search results with AI-generated responses.
- **Product catalog embedding script**: Quickly (re)embeds product catalog data into HANA for semantic search (see `embed_product_catalog.py`).

## Prerequisites

- Python 3.11.x
- SAP HANA Cloud instance
- Cloud Foundry CLI
- Required Python dependencies (see `requirements.txt`)


## Environment Configuration

### 1. HANA Cloud Credentials
- In Cloud Foundry: Provide HANA credentials as a User-Provided Service (UPS). The service name must match the value of `HANA_UPS_NAME` (default: `ups_RMILLERYOUR_NUMBER`).
- Locally: Create an `env_cloud.json` file with your HANA credentials (host, port, user, password, schema).

### 2. AI Core Configuration
- In Cloud Foundry: Bind the managed AI Core service (default: `default_aicore`).
- Locally: Create an `env_config.json` file with your AI Core configuration (API URLs, client ID, client secret, resource group).

### 3. XSUAA Authentication
- The deployment script (`deploy.sh`) will create the required XSUAA service (`product_search_rag-python-srv-uaa`) and service key if not present.

### 4. Environment Variables
- Most configuration is handled via environment variables set in `manifest.yml` (model names, top_k, max_tokens, temperature, service names).


## Running Locally

### 1. Start the Flask server

```bash
python3 server.py
```

The service will be accessible at [http://localhost:3000](http://localhost:3000).

### 2. (Re)embed the product catalog into HANA

If you update the product catalog CSV or want to refresh the vector store, run:

```bash
python3 embed_product_catalog.py
```

This will:
- Load the product catalog from `../data/product_catalog.csv`
- Generate embeddings for each product using the configured embedding model
- Store the vectors and metadata in your HANA table (see VECTOR_TABLE in the script)

## Deployment


## Deployment

1. **Deploy to Cloud Foundry**
    - Run the deployment script:

    ```bash
    bash deploy.sh
    ```

    The script will:
    - Ensure the XSUAA service and service key exist (creates if missing)
    - Remind you to create and bind the HANA UPS if not already present
    - Deploy the Flask application using `cf push` with the configuration in `manifest.yml`

2. **Manifest Configuration**
    - Edit `manifest.yml` to set your application name and ensure the correct service bindings:
      - XSUAA service (authentication)
      - AI Core service
      - HANA UPS (user-provided service)

## Endpoints


### `/retrieveData` (POST)
Processes user queries and returns AI-generated responses using RAG (Retrieval-Augmented Generation).

#### Request Body
The request body must include:

```json
{
    "query": "Your question here"
}
```

#### Response
On success:
```json
{
    "result": "AI-generated response"
}
```
On error:
```json
{
    "error": "Error message"
}
```


## Key Files

- `server.py`: Main Flask application for RAG workflow (vector search + LLM response).
- `embed_product_catalog.py`: Script to (re)embed the product catalog CSV into HANA with fresh embeddings. Run manually after updating the catalog or to reset the vector store.
- `deploy.sh`: Automated deployment script for Cloud Foundry, including XSUAA service creation and app deployment.
- `manifest.yml`: Cloud Foundry manifest specifying app name, environment variables, and service bindings.
- `requirements.txt`: Python dependencies for the project.
- `runtime.txt`: Python runtime version for deployment.
- `Procfile`: Command to run the Flask app in the cloud.
- `env_cloud.json`: (Local only) HANA Cloud credentials.
- `env_config.json`: (Local only) AI Core configuration.
- `xs-security.json`: Security configuration for XSUAA.
- `.gitignore`: Files and directories to ignore in Git.
- `README.md`: Project documentation.


## Security

- Authentication is enforced via SAP XSUAA in Cloud Foundry deployments. Local testing does not require authentication.
- Sensitive credentials should be stored securely in `env_cloud.json` and `env_config.json` (local only). Never commit these files to version control.

## License

This project is licensed under the Apache 2.0 License.