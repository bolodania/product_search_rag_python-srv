"""
embed_product_catalog_dg.py
----------------------------
Indexes the product catalog CSV into SAP AI Core Document Grounding
using the VectorAPIClient from the SAP Generative AI Hub SDK.

This is the Document Grounding equivalent of embed_product_catalog.py
which used SAP HANA Cloud as the vector store.

Usage:
    python embed_product_catalog_dg.py

Prerequisites:
    - env_config.json with AI Core credentials
    - data/product_catalog.csv (relative path: ../data/product_catalog.csv)
"""

import csv
import json
import time

from ai_core_sdk.ai_core_v2_client import AICoreV2Client
from gen_ai_hub.proxy.gen_ai_hub_proxy import GenAIHubProxyClient
from gen_ai_hub.document_grounding import (
    VectorAPIClient,
    CollectionCreateRequest,
    EmbeddingConfig,
    DocumentsCreateRequest,
    BaseDocument,
    TextOnlyBaseChunk,
    VectorKeyValueListPair,
)

# ---- CONFIG ----
COLLECTION_TITLE = "products-it-accessories"
CSV_PATH = "../data/product_catalog.csv"
EMBEDDING_MODEL = "text-embedding-3-large"
BATCH_SIZE = 10

CONTENT_COLUMNS = [
    "PRODUCT_NAME", "DESCRIPTION", "UNIT_PRICE", "LEAD_TIME_DAYS",
    "STOCK_QUANTITY", "RATING", "MIN_ORDER", "CATEGORY", "SUPPLIER_NAME",
    "SUPPLIER_COUNTRY", "SUPPLIER_CITY", "SUPPLIER_ADDRESS", "STATUS",
    "CURRENCY", "MANUFACTURER", "CITY_LAT", "CITY_LONG",
]
METADATA_COLUMNS = [
    "SUPPLIER_ID", "CATEGORY", "SUPPLIER_COUNTRY", "SUPPLIER_CITY", "MANUFACTURER",
]


# ---- CSV LOADER ----
class Document:
    """Minimal document container (replaces LangChain Document)."""
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


def load_csv_documents(file_path: str) -> list:
    """Load CSV rows as Document objects using the built-in csv module."""
    docs = []
    with open(file_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";", quotechar='"')
        for row in reader:
            page_content = "\n".join(
                f"{col}: {row[col]}" for col in CONTENT_COLUMNS if col in row
            )
            metadata = {col: row[col] for col in METADATA_COLUMNS if col in row}
            metadata["source"] = row.get("PRODUCT_ID", "")
            docs.append(Document(page_content=page_content, metadata=metadata))
    return docs


# ---- COLLECTION ----
def create_collection(vector_client: VectorAPIClient, title: str, embedding_model: str) -> str:
    """
    Create a Document Grounding collection (async).

    Uses a before/after snapshot of collection IDs to reliably identify the
    newly created collection — avoids depending on the Location header or
    JSON response body (the API returns 202 Accepted with an empty body).

    Polls get_collection_creation_status() until the collection is CREATED.
    Returns the collection ID.
    """
    # Snapshot existing collection IDs before creation
    before_ids = {c.id for c in vector_client.get_collections().resources}

    try:
        vector_client.create_collection(
            CollectionCreateRequest(
                title=title,
                embeddingConfig=EmbeddingConfig(modelName=embedding_model),
            )
        )
    except Exception as e:
        if "Expecting value" not in str(e):
            raise  # only swallow the empty-body 202 JSON parse error

    # Wait briefly then find the new collection (not in before_ids)
    time.sleep(5)
    after = vector_client.get_collections().resources
    new_cols = [c for c in after if c.id not in before_ids and c.title == title]
    if not new_cols:
        raise RuntimeError(f"New collection '{title}' not found after creation")

    collection_id = new_cols[0].id
    print(f"   Collection ID: {collection_id}")
    print(f"   Polling creation status...")

    # Poll until CREATED
    for attempt in range(30):
        status_resp = vector_client.get_collection_creation_status(collection_id)
        current_status = status_resp.status if hasattr(status_resp, "status") else "CREATED"
        if current_status == "CREATED":
            print(f"   Collection ready (attempt {attempt + 1})")
            return collection_id
        print(f"   Attempt {attempt + 1}: status={current_status}, retrying...")
        time.sleep(2)

    raise TimeoutError("Collection creation timed out after 60 seconds")


# ---- UPLOAD ----
def upload_batch(vector_client: VectorAPIClient, collection_id: str, docs: list) -> None:
    """Upload a batch of Document objects to the Document Grounding collection."""
    vector_client.create_documents(
        collection_id,
        DocumentsCreateRequest(
            documents=[
                BaseDocument(
                    metadata=[
                        VectorKeyValueListPair(key=k, value=[str(v)])
                        for k, v in doc.metadata.items()
                        if v is not None
                    ],
                    chunks=[TextOnlyBaseChunk(content=doc.page_content, metadata=[])],
                )
                for doc in docs
            ]
        ),
    )


def delete_existing_documents(vector_client: VectorAPIClient, collection_id: str) -> None:
    """
    Delete all documents in the collection before re-uploading.

    Ensures a clean state on re-run without needing to recreate the collection.
    The delete_document() API returns 204 No Content (empty body); the SDK
    raises a JSONDecodeError when parsing the empty response — we swallow that
    specific error since the deletion itself succeeded.
    """
    existing = vector_client.get_documents(collection_id, top=1)
    if len(existing.resources) == 0:
        return

    print(f"   Deleting existing documents from collection (clean re-index)...")
    all_docs = vector_client.get_documents(collection_id, top=1000)
    for doc in all_docs.resources:
        try:
            vector_client.delete_document(collection_id, doc.id)
        except Exception as e:
            if "Expecting value" not in str(e):
                raise  # only swallow the empty-body 204 parse error
    print(f"   Deleted {len(all_docs.resources)} documents")


# ---- MAIN ----
def main():
    start_time = time.time()

    # Load AI Core config
    with open("env_config.json") as f:
        aicore_config = json.load(f)

    # Initialize AI Core + GenAI Hub clients
    print("Initializing SAP AI Core client...")
    ai_core_client = AICoreV2Client(
        base_url=aicore_config["AICORE_BASE_URL"],
        auth_url=aicore_config["AICORE_AUTH_URL"],
        client_id=aicore_config["AICORE_CLIENT_ID"],
        client_secret=aicore_config["AICORE_CLIENT_SECRET"],
        resource_group=aicore_config["AICORE_RESOURCE_GROUP"],
    )
    proxy_client = GenAIHubProxyClient(ai_core_client=ai_core_client)
    vector_client = VectorAPIClient(proxy_client=proxy_client)
    print("✅ VectorAPIClient initialized")

    # Load CSV
    print(f"Loading CSV: {CSV_PATH}")
    docs = load_csv_documents(CSV_PATH)
    print(f"Loaded {len(docs)} documents")

    # Create new collection
    print(f"Creating collection '{COLLECTION_TITLE}'...")
    collection_id = create_collection(vector_client, COLLECTION_TITLE, EMBEDDING_MODEL)
    print(f"Collection created: {collection_id}")

    # Delete any existing documents (clean re-index on re-run)
    delete_existing_documents(vector_client, collection_id)

    # Upload documents in batches
    print(f"Uploading {len(docs)} documents in batches of {BATCH_SIZE}...")
    total_uploaded = 0
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        upload_batch(vector_client, collection_id, batch)
        total_uploaded += len(batch)
        print(f"   Uploaded {total_uploaded}/{len(docs)} documents")

    print(f"✅ Collection '{COLLECTION_TITLE}' (ID: {collection_id}) is ready.")
    print(f"   Total execution time: {time.time() - start_time:.2f}s")
    print()
    print(f"Update DG_COLLECTION_ID in manifest.yml to: {collection_id}")


if __name__ == "__main__":
    main()