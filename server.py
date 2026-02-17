import os
import json
import logging
import functools
from typing import Tuple, List, Dict, Any

from flask import Flask, request, jsonify
from hdbcli import dbapi
from cfenv import AppEnv
from sap import xssec

from gen_ai_hub.proxy.gen_ai_hub_proxy import GenAIHubProxyClient
from ai_core_sdk.ai_core_v2_client import AICoreV2Client
from gen_ai_hub.proxy.langchain.init_models import init_llm, init_embedding_model

from langchain_hana import HanaDB
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
# Configuration Loader (VCAP/UVPs or Local JSON fallback)
# ==========================================================
env = AppEnv()


# ------------------------------
# HANA configuration (UPVs)
# ------------------------------
def load_hana_config():
    """
    Load HANA credentials from environment variables (UPVs) for CF deployment,
    or fall back to local env_cloud.json for local development.
    """
    # 1. Check if required env vars exist
    required_vars = ["HANA_HOST", "HANA_USER", "HANA_PASSWORD"]
    if all(os.getenv(v) for v in required_vars):
        logger.info("Using HANA credentials from environment variables (UPVs)")
        return {
            "url": os.getenv("HANA_HOST"),
            "port": int(os.getenv("HANA_PORT", "443")),
            "user": os.getenv("HANA_USER"),
            "pwd": os.getenv("HANA_PASSWORD"),
            "schema": os.getenv("HANA_SCHEMA"),  # optional
        }

    # 2. Fallback to local JSON
    if os.path.exists("env_cloud.json"):
        logger.info("Using local HANA config from env_cloud.json")
        with open("env_cloud.json") as f:
            return json.load(f)

    # 3. No credentials found → fail fast
    raise RuntimeError(
        "No HANA credentials found. "
        "Set HANA_HOST, HANA_USER, HANA_PASSWORD (and optionally HANA_SCHEMA) "
        "or provide env_cloud.json locally."
    )


# Load config
hana_env = load_hana_config()


def get_service_credentials(label=None, name=None):
    """
    Returns credentials from CF service binding.
    Supports lookup by label or service name.
    """
    try:
        if name:
            return env.get_service(name=name).credentials
        if label:
            services = env.get_services(label=label)
            if services:
                return list(services.values())[0].credentials
    except Exception:
        pass
    return None


# ------------------------------
# AI Core configuration
# ------------------------------
aicore_creds = get_service_credentials(label="aicore")

if aicore_creds:
    logger.info("Using AI Core credentials from VCAP_SERVICES")

    aicore_cfg = {
        "AICORE_BASE_URL": aicore_creds.get("base_url")
        or aicore_creds.get("serviceurls", {}).get("AI_API_URL"),
        "AICORE_AUTH_URL": aicore_creds.get("url")
        or aicore_creds.get("uaa", {}).get("url"),
        "AICORE_CLIENT_ID": aicore_creds.get("clientid"),
        "AICORE_CLIENT_SECRET": aicore_creds.get("clientsecret"),
        "AICORE_RESOURCE_GROUP": aicore_creds.get("resource_group"),
    }

else:
    logger.info("Using local AI Core config (env_config.json)")
    with open("env_config.json") as f:
        aicore_cfg = json.load(f)


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

VECTOR_TABLE = f"PRODUCTS_IT_ACCESSORY_OPENAI_{hana_env['user']}"
TOP_K = int(os.getenv("TOP_K", 15))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-large")
embedding_model = init_embedding_model(EMBEDDING_MODEL_NAME, proxy_client=proxy_client)

CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "anthropic--claude-4.5-sonnet")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 800))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))

llm = init_llm(
    CHAT_MODEL_NAME,
    proxy_client=proxy_client,
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
)

# ==========================================================
# Flask App
# ==========================================================
app = Flask(__name__)


# ==========================================================
# Helper — Build Vector Store
# ==========================================================
def get_vector_store(conn):
    return HanaDB(
        embedding=embedding_model,
        connection=conn,
        table_name=VECTOR_TABLE,
        content_column="VEC_TEXT",
        metadata_column="VEC_META",
        vector_column="VEC_VECTOR",
    )


# ==========================================================
# Helper — Generate RAG answer
# ==========================================================
def generate_rag_response(query: str, hana_conn) -> str:

    db = get_vector_store(hana_conn)

    results = db.similarity_search_with_score(query=query, k=TOP_K)

    context = "\n\n".join(doc.page_content for doc, _ in results)

    SYSTEM_RAG_PROMPT = """
    You are a product recommendation assistant.

    Your job is to help users find and understand products using ONLY the information provided in the retrieved context.

    You must follow these rules strictly:

    --------------------------------------------------
    1. Use only retrieved context
    --------------------------------------------------
    - Base your answer only on the provided context.
    - Never invent product names, specifications, ratings, or prices.
    - If required information is missing, say you don’t know.

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
    "I don’t have enough information to answer that."

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

    PROMPT = PromptTemplate(
        template=SYSTEM_RAG_PROMPT, input_variables=["query", "context"]
    )

    chain = PROMPT | llm | StrOutputParser()

    response = chain.invoke({"query": query, "context": context})

    return response.strip()


# ==========================================================
# Helper — Main processing logic
# ==========================================================
def process_query(data: Dict[str, Any], hana_conn):

    query = data["query"]

    response = generate_rag_response(
        query=query,
        hana_conn=hana_conn,
    )

    # try to parse JSON automatically if model returned JSON
    try:
        return json.loads(response)
    except Exception:
        return response


# ==========================================================
# AUTH
# ==========================================================
uaa_service = None

if not local_testing:
    uaa_service = get_service_credentials(label="xsuaa")


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

    hana_conn = None

    try:
        data = request.get_json()

        required = ["query"]
        if not data or any(f not in data for f in required):
            return jsonify({"error": "Invalid JSON input"}), 400

        hana_conn = dbapi.connect(
            address=hana_env["url"],
            port=hana_env["port"],
            user=hana_env["user"],
            password=hana_env["pwd"],
        )

        result = process_query(data, hana_conn)

        if not result:
            return jsonify({"error": "No result"}), 400

        return jsonify({"result": result}), 200

    except Exception as e:
        logger.exception("Error in /retrieveData")
        return jsonify({"error": str(e)}), 500

    finally:
        if hana_conn:
            try:
                hana_conn.close()
            except:
                pass


# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 3000)))
