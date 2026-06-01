"""
server_dg.py
-------------
Flask server using SAP AI Core Document Grounding + Orchestration Service V2 for RAG.

This is the Document Grounding equivalent of server.py which used
SAP HANA Cloud as the vector store.

The Orchestration Service handles retrieval and generation in a single call:
  - GroundingModule performs semantic search against the Document Grounding collection
  - LLM generates the answer using the retrieved context
  - No manual vector search or prompt building required

Environment variables:
    DG_COLLECTION_ID        Document Grounding collection ID (required).
                            Obtain this by running embed_product_catalog_dg.py.
    DG_COLLECTION_TITLE     Collection title used to auto-resolve the ID at
                            startup if DG_COLLECTION_ID is not set.
                            Default: "products-it-accessories"
    TOP_K                   Number of chunks to retrieve per query. Default: 15
    CHAT_MODEL_NAME         LLM model name. Default: gpt-4o-mini
    MAX_TOKENS              LLM max tokens. Default: 800
    TEMPERATURE             LLM temperature. Default: 0.3
    AI_CORE_RESOURCE_GROUP  AI Core resource group. Default: default
    AI_CORE_INSTANCE_NAME   CF service binding name for AI Core. Default: default_aicore
    XSUAA_SERVICE_NAME      CF XSUAA binding name (production only).
"""

import os
import json
import logging
import functools
from typing import Dict, Any

from flask import Flask, request, jsonify
from cfenv import AppEnv
from sap import xssec

from ai_core_sdk.ai_core_v2_client import AICoreV2Client
from gen_ai_hub.proxy.gen_ai_hub_proxy import GenAIHubProxyClient
from gen_ai_hub.document_grounding import VectorAPIClient

from gen_ai_hub.orchestration_v2 import (
    OrchestrationService,
    OrchestrationConfig,
    ModuleConfig,
    LLMModelDetails,
    PromptTemplatingModuleConfig,
    Template,
    SystemMessage,
    UserMessage,
    GroundingModuleConfig,
    GroundingType,
    DocumentGroundingConfig,
    DocumentGroundingPlaceholders,
    DocumentGroundingFilter,
    DataRepositoryType,
    GroundingSearchConfig,
)

# ==========================================================
# Logging
# ==========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================================
# Local testing flag
# ==========================================================
local_testing = not bool(os.getenv("VCAP_SERVICES"))

# ==========================================================
# Configuration Loader (VCAP or Local JSON fallback)
# ==========================================================
env = AppEnv()

# ------------------------------
# AI Core configuration
# ------------------------------
AI_CORE_RESOURCE_GROUP = os.getenv("AI_CORE_RESOURCE_GROUP", "default")
AI_CORE_INSTANCE_NAME = os.getenv("AI_CORE_INSTANCE_NAME", "default_aicore")

aicore_creds = (
    env.get_service(name=AI_CORE_INSTANCE_NAME).credentials
    if env.get_service(name=AI_CORE_INSTANCE_NAME)
    else None
)

if aicore_creds:
    logger.info(f"Using AI Core credentials from service: {AI_CORE_INSTANCE_NAME}")
    aicore_cfg = {
        "AICORE_BASE_URL": aicore_creds["serviceurls"]["AI_API_URL"] + "/v2",
        "AICORE_AUTH_URL": aicore_creds["url"],
        "AICORE_CLIENT_ID": aicore_creds["clientid"],
        "AICORE_CLIENT_SECRET": aicore_creds["clientsecret"],
        "AICORE_RESOURCE_GROUP": AI_CORE_RESOURCE_GROUP,
    }
else:
    logger.warning("AI Core service not found, trying env_config.json")
    try:
        with open("env_config.json") as f:
            aicore_cfg = json.load(f)
    except Exception as e:
        raise RuntimeError("AI Core config not available from CF or local file") from e

if not aicore_cfg:
    raise RuntimeError("AI Core config is empty or None")

# ==========================================================
# Initialize SAP AI Core + GenAI Hub
# ==========================================================
ai_core_client = AICoreV2Client(
    base_url=aicore_cfg["AICORE_BASE_URL"],
    auth_url=aicore_cfg["AICORE_AUTH_URL"],
    client_id=aicore_cfg["AICORE_CLIENT_ID"],
    client_secret=aicore_cfg["AICORE_CLIENT_SECRET"],
    resource_group=aicore_cfg["AICORE_RESOURCE_GROUP"],
)

proxy_client = GenAIHubProxyClient(ai_core_client=ai_core_client)
vector_client = VectorAPIClient(proxy_client=proxy_client)
logger.info("AI Core client initialized successfully")

# ==========================================================
# Orchestration Service setup (V2)
# ==========================================================
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "gpt-4o-mini")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 800))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))
TOP_K = int(os.getenv("TOP_K", 15))

# Discover the running Orchestration Service deployment
try:
    orchestration_deployment = next(
        d for d in ai_core_client.deployment.query().resources
        if "orchestration" in d.scenario_id.lower() and d.status.value == "RUNNING"
    )
    orchestration_url = (
        f"{aicore_cfg['AICORE_BASE_URL']}/inference/deployments/{orchestration_deployment.id}"
    )
    orchestration_service = OrchestrationService(
        api_url=orchestration_url,
        proxy_client=proxy_client,
    )
    logger.info(f"Orchestration Service (V2) initialized. Deployment ID: {orchestration_deployment.id}")
except StopIteration:
    raise RuntimeError(
        "Orchestration Service deployment not found. "
        "Ensure the Orchestration Service is enabled in your AI Core instance."
    )

# ==========================================================
# Document Grounding — resolve collection ID
# ==========================================================
DG_COLLECTION_TITLE = os.getenv("DG_COLLECTION_TITLE", "products-it-accessories")


def _resolve_collection_id() -> str:
    """
    Resolve the Document Grounding collection ID.

    Priority:
    1. DG_COLLECTION_ID environment variable (fastest — set after running embed script)
    2. Auto-lookup by DG_COLLECTION_TITLE via VectorAPIClient.get_collections()
    """
    col_id = os.getenv("DG_COLLECTION_ID", "").strip()
    if col_id:
        logger.info(f"Using DG_COLLECTION_ID from environment: {col_id}")
        return col_id

    logger.info(f"DG_COLLECTION_ID not set, resolving by title: '{DG_COLLECTION_TITLE}'")
    collections_resp = vector_client.get_collections()
    for col in collections_resp.resources:
        if col.title == DG_COLLECTION_TITLE:
            logger.info(f"Resolved collection '{DG_COLLECTION_TITLE}' -> ID: {col.id}")
            return col.id

    raise RuntimeError(
        f"Document Grounding collection '{DG_COLLECTION_TITLE}' not found.\n"
        "Run embed_product_catalog_dg.py first, then set DG_COLLECTION_ID."
    )


# Resolve collection ID at startup
DG_COLLECTION_ID = _resolve_collection_id()
logger.info(f"Document Grounding ready. Collection ID: {DG_COLLECTION_ID}")

# ==========================================================
# Build Orchestration RAG config (V2)
# ==========================================================
PRODUCT_SYSTEM_PROMPT = (
    "You are a product recommendation assistant.\n\n"
    "Your job is to help users find and understand products using ONLY the information "
    "provided in the retrieved context.\n\n"
    "Rules:\n"
    "1. Base your answer only on the provided context. Never invent product names, "
    "specifications, ratings, or prices.\n"
    "2. If matching products exist: list product name and relevant attributes, explain "
    "briefly why each product matches.\n"
    "3. If no products match, respond exactly: "
    "'I could not find any products that match your criteria.'\n"
    "4. If the context does not contain enough information, respond exactly: "
    "'I don't have enough information to answer that.'\n"
    "5. Do not mention embeddings, vector search, retrieval, metadata, or internal processing.\n"
    "6. Be clear, concise, and factual."
)

rag_config = OrchestrationConfig(
    modules=ModuleConfig(
        prompt_templating=PromptTemplatingModuleConfig(
            prompt=Template(template=[
                SystemMessage(content=PRODUCT_SYSTEM_PROMPT),
                UserMessage(
                    content=(
                        "Context: {{?grounding_response}}\n\n"
                        "User question: {{?question}}"
                    )
                ),
            ]),
            model=LLMModelDetails(
                name=CHAT_MODEL_NAME,
                params={"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS},
            ),
        ),
        grounding=GroundingModuleConfig(
            type=GroundingType.DOCUMENT_GROUNDING_SERVICE,
            config=DocumentGroundingConfig(
                filters=[DocumentGroundingFilter(
                    id="filter-1",
                    data_repositories=[DG_COLLECTION_ID],
                    search_config=GroundingSearchConfig(max_chunk_count=TOP_K),
                    data_repository_type=DataRepositoryType.VECTOR,
                )],
                placeholders=DocumentGroundingPlaceholders(
                    input=["question"],
                    output="grounding_response",
                ),
            ),
        ),
    )
)
logger.info(f"RAG config ready. Model: {CHAT_MODEL_NAME}, TOP_K: {TOP_K}")

# ==========================================================
# Flask App
# ==========================================================
app = Flask(__name__)


# ==========================================================
# Helper — Generate RAG answer via Orchestration Service V2
# ==========================================================
def generate_rag_response(query: str) -> str:
    """
    Use the Orchestration Service to retrieve relevant chunks from Document Grounding
    and generate an LLM response in a single API call.
    """
    response = orchestration_service.run(
        config=rag_config,
        placeholder_values={"question": query},
    )
    return response.final_result.choices[0].message.content.strip()


# ==========================================================
# Helper — Main processing logic
# ==========================================================
def process_query(data: Dict[str, Any]) -> Any:
    query = data["query"]
    response = generate_rag_response(query=query)

    # Try to parse JSON automatically if model returned JSON
    try:
        return json.loads(response)
    except Exception:
        return response


# ==========================================================
# AUTH
# ==========================================================
if not local_testing:
    XSUAA_SERVICE_NAME = os.getenv(
        "XSUAA_SERVICE_NAME", "product_search_rag-python-srv-uaa"
    )
    uaa_service = env.get_service(name=XSUAA_SERVICE_NAME).credentials
    if not uaa_service:
        raise RuntimeError("XSUAA service binding not found")


def require_auth(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not local_testing:
            if "authorization" not in request.headers:
                return jsonify({"error": "Unauthorized"}), 403

            access_token = request.headers.get("authorization")[7:]
            security_context = xssec.create_security_context(access_token, uaa_service)

            if not security_context.check_scope("uaa.resource"):
                return jsonify({"error": "Unauthorized"}), 403

        return f(*args, **kwargs)

    return decorated


# ==========================================================
# REST Endpoint
# ==========================================================
@app.route("/retrieveData", methods=["POST"])
@require_auth
def retrieve_data():
    try:
        data = request.get_json()

        if not data or "query" not in data:
            return jsonify({"error": "Invalid JSON input"}), 400

        result = process_query(data)

        if not result:
            return jsonify({"error": "No result"}), 400

        return jsonify({"result": result}), 200

    except Exception as e:
        logger.exception("Error in /retrieveData")
        return jsonify({"error": str(e)}), 500


# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 3000)))