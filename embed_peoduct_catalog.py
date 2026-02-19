from hdbcli import dbapi
import json
import time
from ai_core_sdk.ai_core_v2_client import AICoreV2Client
from gen_ai_hub.proxy.langchain.init_models import init_embedding_model
from gen_ai_hub.proxy.gen_ai_hub_proxy import GenAIHubProxyClient
from langchain_hana import HanaDB
from langchain_community.document_loaders import CSVLoader

# ---- CONFIG ----
VECTOR_TABLE = "PRODUCTS_IT_ACCESSORY_OPENAI_"
CSV_PATH = "../data/product_catalog.csv"


# ---- MAIN ----
def main():
    start_time = time.time()

    with open("env_cloud.json") as f:
        hana_env = json.load(f)

    with open("env_config.json") as f:
        aicore_config = json.load(f)

    conn = dbapi.connect(
        address=hana_env["url"],
        port=hana_env["port"],
        user=hana_env["user"],
        password=hana_env["pwd"],
    )
    cursor = conn.cursor()

    print("🔄 Loading CSV using CSVLoader...")

    # Process CSV data file
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
    print(f"📄 Loaded {len(docs)} documents")

    # Embedding model
    proxy_client = GenAIHubProxyClient(
        ai_core_client=AICoreV2Client(
            base_url=aicore_config["AICORE_BASE_URL"],
            auth_url=aicore_config["AICORE_AUTH_URL"],
            client_id=aicore_config["AICORE_CLIENT_ID"],
            client_secret=aicore_config["AICORE_CLIENT_SECRET"],
            resource_group=aicore_config["AICORE_RESOURCE_GROUP"],
        )
    )

    embeddings = init_embedding_model(
        "text-embedding-3-large", proxy_client=proxy_client
    )

    # HANA Vector Store
    db = HanaDB(
        connection=conn,
        embedding=embeddings,
        table_name=VECTOR_TABLE + hana_env["user"],
        content_column="VEC_TEXT",  # the original text description of the product details
        metadata_column="VEC_META",  # metadata associated with the product details
        vector_column="VEC_VECTOR",  # the vector representation of each product
    )

    print("🗑 Deleting existing embeddings...")
    db.delete(filter={})

    print("📥 Inserting new embeddings...")
    BATCH_SIZE = 1000

    for i in range(0, len(docs), BATCH_SIZE):
        db.add_documents(docs[i : i + BATCH_SIZE])

    conn.commit()
    cursor.close()
    conn.close()

    print(f"✅ Embedding table refreshed: {VECTOR_TABLE}")
    print(f"⏱ Total execution time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
