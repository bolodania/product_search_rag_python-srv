"""
embed_product_catalog_dg.py
----------------------------
Indexes the product catalog CSV into SAP AI Core Document Grounding
using the Vector API directly (Option 2).

This is the Document Grounding equivalent of embed_product_catalog.py
which used SAP HANA Cloud as the vector store.

Usage:
    python embed_product_catalog_dg.py

Prerequisites:
    - env_config.json with AI Core credentials
    - data/product_catalog.csv (relative path: ../data/product_catalog.csv)
"""

import json
import time
import requests
from langchain_community.document_loaders import CSVLoader

# ---- CONFIG ----
COLLECTION_TITLE = "products-it-accessories"
CSV_PATH = "../data/product_catalog.csv"
EMBEDDING_MODEL = "text-embedding-3-large"
BATCH_SIZE = 10


# ---- AUTH ----
def get_access_token(auth_url: str, client_id: str, client_secret: str) -> str:
    """Retrieve an OAuth2 client credentials access token."""
    response = requests.post(
        f"{auth_url}/oauth/token",
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
    )
    response.raise_for_status()
    return response.json()["access_token"]


# ---- COLLECTION ----
def create_collection(dg_base_url: str, aicore_base_url: str, headers: dict, title: str, embedding_model: str) -> str:
    """
    Create a Document Grounding collection (async).
    Polls the Location URL until the collection is ready.
    Returns the collection ID.
    """
    response = requests.post(
        f"{dg_base_url}/vector/collections",
        headers=headers,
        json={
            "title": title,
            "embeddingConfig": {"modelName": embedding_model},
        },
    )
    response.raise_for_status()  # Expects 202 Accepted

    # Extract collection ID from Location header
    # Format: .../vector/collections/{id}/creationStatus
    location = response.headers.get("Location", "")
    collection_id = location.split("/collections/")[-1].split("/")[0]

    if not collection_id:
        raise RuntimeError(f"Could not extract collection ID from Location header: {location}")

    print(f"   Collection ID: {collection_id}")
    print(f"   Polling creation status...")

    # Build full status URL.
    # Location is relative to AICORE_BASE_URL (not dg_base_url), e.g.:
    #   /lm/document-grounding/vector/collections/{id}/creationStatus
    status_url = location if location.startswith("http") else f"{aicore_base_url}{location}"

    for attempt in range(30):
        status_resp = requests.get(status_url, headers=headers)

        # 204 No Content or empty body = ready
        if status_resp.status_code == 204 or not status_resp.text.strip():
            print(f"   Collection ready (attempt {attempt + 1})")
            return collection_id

        try:
            status_data = status_resp.json()
            current_status = status_data.get("status", "pending")
            if current_status == "created":
                print(f"   Collection ready (status=created, attempt {attempt + 1})")
                return collection_id
            print(f"   Attempt {attempt + 1}: status={current_status}, retrying...")
        except Exception:
            print(f"   Attempt {attempt + 1}: unexpected response ({status_resp.status_code}), retrying...")

        time.sleep(2)

    raise TimeoutError("Collection creation timed out after 60 seconds")


def delete_collection_if_exists(dg_base_url: str, headers: dict, title: str) -> None:
    """Delete an existing collection with the given title (if found)."""
    list_resp = requests.get(f"{dg_base_url}/vector/collections", headers=headers)
    list_resp.raise_for_status()
    collections = list_resp.json().get("resources", [])

    for col in collections:
        if col.get("title") == title:
            col_id = col.get("id")
            print(f"   Found existing collection '{title}' (ID: {col_id}), deleting...")
            del_resp = requests.delete(
                f"{dg_base_url}/vector/collections/{col_id}",
                headers=headers,
            )
            del_resp.raise_for_status()
            print(f"   Deletion initiated for collection ID: {col_id}")
            time.sleep(3)  # Brief wait for deletion to propagate
            return

    print(f"   No existing collection named '{title}' found, skipping delete.")


# ---- UPLOAD ----
def upload_batch(dg_base_url: str, headers: dict, collection_id: str, docs: list) -> None:
    """Upload a batch of LangChain documents to the Document Grounding collection."""
    dg_documents = []
    for doc in docs:
        doc_metadata = [
            {"key": k, "value": [str(v)]}
            for k, v in doc.metadata.items()
            if v is not None
        ]
        dg_documents.append({
            "metadata": doc_metadata,
            "chunks": [
                {
                    "content": doc.page_content,
                    "metadata": [],
                }
            ],
        })

    response = requests.post(
        f"{dg_base_url}/vector/collections/{collection_id}/documents",
        headers=headers,
        json={"documents": dg_documents},
    )
    response.raise_for_status()


# ---- MAIN ----
def main():
    start_time = time.time()

    # Load AI Core config
    with open("env_config.json") as f:
        aicore_config = json.load(f)

    dg_base_url = f"{aicore_config['AICORE_BASE_URL']}/lm/document-grounding"

    # Obtain access token
    print("Authenticating with SAP AI Core...")
    access_token = get_access_token(
        aicore_config["AICORE_AUTH_URL"],
        aicore_config["AICORE_CLIENT_ID"],
        aicore_config["AICORE_CLIENT_SECRET"],
    )

    headers = {
        "Authorization": f"Bearer {access_token}",
        "AI-Resource-Group": aicore_config["AICORE_RESOURCE_GROUP"],
        "Content-Type": "application/json",
    }

    print(f"Document Grounding endpoint: {dg_base_url}")

    # Load CSV
    print(f"Loading CSV: {CSV_PATH}")
    loader = CSVLoader(
        file_path=CSV_PATH,
        source_column="PRODUCT_ID",
        metadata_columns=[
            "SUPPLIER_ID",
            "CATEGORY",
            "SUPPLIER_COUNTRY",
            "SUPPLIER_CITY",
            "MANUFACTURER",
        ],
        content_columns=[
            "PRODUCT_NAME",
            "DESCRIPTION",
            "UNIT_PRICE",
            "LEAD_TIME_DAYS",
            "STOCK_QUANTITY",
            "RATING",
            "MIN_ORDER",
            "CATEGORY",
            "SUPPLIER_NAME",
            "SUPPLIER_COUNTRY",
            "SUPPLIER_CITY",
            "SUPPLIER_ADDRESS",
            "STATUS",
            "CURRENCY",
            "MANUFACTURER",
            "CITY_LAT",
            "CITY_LONG",
        ],
        csv_args={"delimiter": ";", "quotechar": '"'},
        encoding="utf-8-sig",
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")

    # Delete existing collection (clean re-index)
    # print(f"Checking for existing collection '{COLLECTION_TITLE}'...")
    # delete_collection_if_exists(dg_base_url, headers, COLLECTION_TITLE)

    # Create new collection
    print(f"Creating collection '{COLLECTION_TITLE}'...")
    collection_id = create_collection(dg_base_url, aicore_config["AICORE_BASE_URL"], headers, COLLECTION_TITLE, EMBEDDING_MODEL)
    print(f"Collection created: {collection_id}")

    # Upload documents in batches
    print(f"Uploading {len(docs)} documents in batches of {BATCH_SIZE}...")
    total_uploaded = 0
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i: i + BATCH_SIZE]
        upload_batch(dg_base_url, headers, collection_id, batch)
        total_uploaded += len(batch)
        print(f"   Uploaded {total_uploaded}/{len(docs)} documents")

    print(f"Collection '{COLLECTION_TITLE}' (ID: {collection_id}) is ready.")
    print(f"Total execution time: {time.time() - start_time:.2f}s")
    print()
    print(f"Update DG_COLLECTION_ID in manifest.yml to: {collection_id}")


if __name__ == "__main__":
    main()