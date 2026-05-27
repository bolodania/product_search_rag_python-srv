"""
server_dg.py
-------------
Flask server using SAP AI Core Document Grounding (Vector API) for RAG.

This is the Document Grounding equivalent of server.py which used
SAP HANA Cloud as the vector store.

Environment variables:
    DG_COLLECTION_ID        Document Grounding collection ID (required).
                            Obtain this by running embed_product_catalog_dg.py.
    DG_COLLECTION_TITLE     Collection title used to auto-resolve the ID at
                            startup if DG_COLLECTION_ID is not set.
                            Default: "products-it-accessories"
    TOP_K                   Number of chunks to retrieve per query. Default: 15
    CHAT_MODEL_NAME         LLM model name. Default: anthropic--claude-4.5-sonnet
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
import time
from typing import Dict, Any

import requests
from flask import Flask, request, jsonify
from cfenv import AppEnv
from sap import xssec

from gen_ai_hub.proxy.gen_ai_hub_proxy import GenAIHubProxyClient
from ai_core_sdk.ai_core_v2_client import AICoreV2Client
from gen_ai_hub.proxy.langchain.init_models import init_llm

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

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
        "AICORE_AUTH_URL": aicore_creds["url"] + "/oauth/token",
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

CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "anthropic--claude-4.5-sonnet")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 800))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))
TOP_K = int(os.getenv("TOP_K", 15))

llm = init_llm(
    CHAT_MODEL_NAME,
    proxy_client=proxy_client,
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
    top_p=None,
)

# ==========================================================
# Document Grounding setup
# ==========================================================
DG_BASE_URL = f"{aicore_cfg['AICORE_BASE_URL']}/lm/document-grounding"
DG_COLLECTION_TITLE = os.getenv("DG_COLLECTION_TITLE", "products-it-accessories")


def _get_dg_access_token() -> str:
    """Obtain a fresh OAuth2 token for Document Grounding API calls."""
    response = requests.post(
        f"{aicore_cfg['AICORE_AUTH_URL']}/oauth/token",
        data={
            "grant_type": "client_credentials",
            "client_id": aicore_cfg["AICORE_CLIENT_ID"],
            "client_secret": aicore_cfg["AICORE_CLIENT_SECRET"],
        },
    )
    response.raise_for_status()
    return response.json()["access_token"]


def _dg_headers() -> dict:
    """Build HTTP headers for Document Grounding API calls with a fresh token."""
    return {
        "Authorization": f"Bearer {_get_dg_access_token()}",
        "AI-Resource-Group": aicore_cfg["AICORE_RESOURCE_GROUP"],
        "Content-Type": "application/json",
    }


def _resolve_collection_id() -> str:
    """
    Resolve the Document Grounding collection ID.

    Priority:
    1. DG_COLLECTION_ID environment variable (fastest — set after running embed script)
    2. Auto-lookup by DG_COLLECTION_TITLE via GET /vector/collections
    """
    # 1. Explicit env var
    col_id = os.getenv("DG_COLLECTION_ID", "").strip()
    if col_id:
        logger.info(f"Using DG_COLLECTION_ID from environment: {col_id}")
        return col_id

    # 2. Auto-resolve by title
    logger.info(f"DG_COLLECTION_ID not set, resolving by title: '{DG_COLLECTION_TITLE}'")
    resp = requests.get(f"{DG_BASE_URL}/vector/collections", headers=_dg_headers())
    resp.raise_for_status()
    collections = resp.json().get("resources", [])

    for col in collections:
        if col.get("title") == DG_COLLECTION_TITLE:
            resolved_id = col["id"]
            logger.info(f"Resolved collection '{DG_COLLECTION_TITLE}' -> ID: {resolved_id}")
            return resolved_id

    raise RuntimeError(
        f"Document Grounding collection '{DG_COLLECTION_TITLE}' not found.\n"
        "Run embed_product_catalog_dg.py first, then set DG_COLLECTION_ID."
    )


# Resolve collection ID at startup
DG_COLLECTION_ID = _resolve_collection_id()
logger.info(f"Document Grounding ready. Collection ID: {DG_COLLECTION_ID}")

# ==========================================================
# Flask App
# ==========================================================
app = Flask(__name__)


# ==========================================================
# Helper — Semantic search via Document Grounding
# ==========================================================
def search_documents(query: str, top_k: int = TOP_K) -> list:
    """
    Perform semantic search using POST /vector/search.
    Returns a list of chunk content strings.
    """
    response = requests.post(
        f"{DG_BASE_URL}/vector/search",
        headers=_dg_headers(),
        json={
            "query": query,
            "filters": [
                {
                    "id": "filter-1",
                    "collectionIds": [DG_COLLECTION_ID],
                    "configuration": {"maxChunkCount": top_k},
                }
            ],
        },
    )
    response.raise_for_status()
    search_results = response.json()

    # Extract chunk content from nested response structure:
    # results -> VectorPerFilterSearchResult
    #   -> results -> DocumentsChunk
    #     -> documents -> Document-Output
    #       -> chunks -> VectorChunk { id, content, metadata }
    chunks = []
    for filter_result in search_results.get("results", []):
        for collection_result in filter_result.get("results", []):
            for doc in collection_result.get("documents", []):
                for chunk in doc.get("chunks", []):
                    content = chunk.get("content", "")
                    if content:
                        chunks.append(content)
    return chunks


# ==========================================================
# Helper — Generate RAG answer
# ==========================================================
def generate_rag_response(query: str) -> str:
    """Retrieve relevant chunks from Document Grounding and generate an LLM response."""

    chunks = search_documents(query, top_k=TOP_K)
    context = "\n\n".join(chunks)

    PRODUCT_RAG_PROMPT = """
    You are a product recommendation assistant.

    Your job is to help users find and understand products using ONLY the information provided in the retrieved context.

    You must follow these rules strictly:

    --------------------------------------------------
    1. Use only retrieved context
    --------------------------------------------------
    - Base your answer only on the provided context.
    - Never invent product names, specifications, ratings, or prices.
    - If required information is missing, say you don't know.

    --------------------------------------------------
    2. Understand user intent
    --------------------------------------------------
    The user may ask to:
    - recommend products
    - filter products by criteria (rating, price, category, brand, features)
    - compare products
    - explain product features
    - summarize options
    - find best match for a need

    Interpret the request and use the context to respond appropriately.

    --------------------------------------------------
    3. When recommending products
    --------------------------------------------------
    If matching products exist:
    - return only products that meet the criteria
    - clearly list product name and relevant attributes
    - explain briefly why each product matches

    If no products match:
    Respond exactly:
    "I could not find any products that match your criteria."

    --------------------------------------------------
    4. When information is incomplete
    --------------------------------------------------
    If the context does not contain enough information:
    Respond exactly:
    "I don't have enough information to answer that."

    --------------------------------------------------
    5. Do not expose system details
    --------------------------------------------------
    Never mention:
    - embeddings
    - vector search
    - retrieval
    - metadata
    - internal processing

    --------------------------------------------------
    6. Response style
    --------------------------------------------------
    - Be clear and concise
    - Use structured lists when helpful
    - Be factual and neutral
    - Do not speculate

    --------------------------------------------------

    User question:
    {query}

    Context:
    {context}
    """

    human_message_prompt = HumanMessagePromptTemplate.from_template(PRODUCT_RAG_PROMPT)
    chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])

    prompt_text = chat_prompt.format_prompt(query=query, context=context).to_string()
    llm_response = llm.invoke(prompt_text)

    return llm_response.content.strip()


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