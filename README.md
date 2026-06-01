# product_search_rag_YOUR_NUMBER-python-srv

This is the Python-based service for the Product Search RAG application. It provides retrieval-augmented generation (RAG) capabilities using SAP Generative AI Hub models.

This service is available in **two versions**:

| | Document Grounding (default) | HANA Cloud (optional) |
|---|---|---|
| **Vector store** | SAP AI Core Document Grounding | SAP HANA Cloud |
| **Server** | `server_dg.py` | `server.py` |
| **Embed script** | `embed_product_catalog_dg.py` | `embed_product_catalog.py` |
| **Active** | Yes (default) | Optional alternative |

> The **Document Grounding version is used by default** (`Procfile` points to `server_dg.py`). The HANA Cloud version is a fully functional alternative — to use it instead, update `Procfile` to `web: python server.py`, restore the `HANA_UPS_NAME` env var and `ups_RMILLERYOUR_NUMBER` service binding in `manifest.yml`, and run `embed_product_catalog.py` to index the catalog into HANA.


## Features

- Flask-based API for processing user queries and interacting with AI models.
- **SAP AI Core Document Grounding** integration for semantic retrieval using the Vector API.
- Generative AI Hub SDK for chat completions.
- Secure authentication using SAP XSUAA.
- RAG workflow combining Document Grounding search results with AI-generated responses.
- **Product catalog embedding script**: Indexes product catalog data into a Document Grounding collection for semantic search (see `embed_product_catalog_dg.py`).

## Prerequisites

- Python 3.11.x
- SAP AI Core instance with **Document Grounding** capability enabled
- Cloud Foundry CLI
- Required Python dependencies (see `requirements.txt`)


## Environment Configuration

### 1. Document Grounding Collection
- Run `embed_product_catalog_dg.py` once to create the collection and index the product catalog.
- The script will print the collection ID — update `DG_COLLECTION_ID` in `manifest.yml` with this value.
- The collection title defaults to `products-it-accessories`.

### 2. AI Core Configuration
- In Cloud Foundry: Bind the managed AI Core service (default: `default_aicore`).
- Locally: Create an `env_config.json` file with your AI Core configuration (API URLs, client ID, client secret, resource group).

### 3. XSUAA Authentication
- The deployment script (`deploy.sh`) will create the required XSUAA service (`product_search_rag-python-srv-uaa`) and service key if not present.

### 4. Environment Variables
- Most configuration is handled via environment variables set in `manifest.yml`:

| Variable | Description | Default |
|---|---|---|
| `DG_COLLECTION_ID` | Document Grounding collection ID (set after running embed script) | — |
| `DG_COLLECTION_TITLE` | Collection title for auto-resolution if ID not set | `products-it-accessories` |
| `CHAT_MODEL_NAME` | LLM model name | `gpt-4o-mini` |
| `EMBEDDING_MODEL_NAME` | Embedding model name | `text-embedding-3-large` |
| `TOP_K` | Number of chunks to retrieve per query | `15` |
| `MAX_TOKENS` | LLM max tokens | `800` |
| `TEMPERATURE` | LLM temperature | `0.3` |
| `AI_CORE_INSTANCE_NAME` | CF service binding name for AI Core | `default_aicore` |
| `AI_CORE_RESOURCE_GROUP` | AI Core resource group | `default` |


## Running Locally

### 1. Index the product catalog into Document Grounding

Run this once before starting the server (or whenever the catalog changes):

```bash
python3 embed_product_catalog_dg.py
```

This will:
- Authenticate with SAP AI Core using `env_config.json`
- Create a new Document Grounding collection (`products-it-accessories`)
- Upload all product records from `../data/product_catalog.csv` as document chunks
- Print the collection ID — update `DG_COLLECTION_ID` in `manifest.yml` with this value

### 2. Start the Flask server

```bash
export DG_COLLECTION_ID=<your-collection-id>
python3 server_dg.py
```

The service will be accessible at [http://localhost:3000](http://localhost:3000).

> If `DG_COLLECTION_ID` is not set, the server will auto-resolve the collection ID by title at startup.


## Deployment

1. **Index the product catalog** (if not done already):

    ```bash
    python3 embed_product_catalog_dg.py
    ```

    Update `DG_COLLECTION_ID` in `manifest.yml` with the printed collection ID.

2. **Deploy to Cloud Foundry**:

    ```bash
    bash deploy.sh
    ```

    The script will:
    - Ensure the XSUAA service and service key exist (creates if missing)
    - Deploy the Flask application using `cf push` with the configuration in `manifest.yml`

3. **Manifest Configuration**:
    - Edit `manifest.yml` to set your application name and `DG_COLLECTION_ID`.
    - Required service bindings: XSUAA service, AI Core service.


## Endpoints

### `/retrieveData` (POST)
Processes user queries and returns AI-generated responses using RAG (Retrieval-Augmented Generation).

#### Request Body

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

### Document Grounding (default)
- `server_dg.py`: Main Flask application using SAP AI Core Document Grounding for RAG.
- `embed_product_catalog_dg.py`: Script to index the product catalog CSV into a Document Grounding collection. Run manually after updating the catalog.

### HANA Cloud (optional alternative)
- `server.py`: Flask application using SAP HANA Cloud vector search for RAG.
- `embed_product_catalog.py`: Script to embed the product catalog into HANA. Requires a HANA Cloud instance and UPS binding.

### Shared
- `deploy.sh`: Automated deployment script for Cloud Foundry.
- `manifest.yml`: Cloud Foundry manifest specifying app name, environment variables, and service bindings.
- `Procfile`: Command to run the Flask app (`server_dg.py` by default).
- `requirements.txt`: Python dependencies for the project.
- `runtime.txt`: Python runtime version for deployment.
- `env_config.json`: (Local only) AI Core configuration.
- `env_cloud.json`: (Local only) HANA Cloud credentials (only needed for HANA version).
- `xs-security.json`: Security configuration for XSUAA.


## Security

- Authentication is enforced via SAP XSUAA in Cloud Foundry deployments. Local testing does not require authentication.
- Sensitive credentials should be stored securely in `env_cloud.json` and `env_config.json` (local only). Never commit these files to version control.

## License

This project is licensed under the Apache 2.0 License.