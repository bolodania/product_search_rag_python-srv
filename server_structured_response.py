import os
import json
import atexit
from flask import Flask, request, jsonify
from hdbcli import dbapi
from cfenv import AppEnv
from sap import xssec
import functools
from gen_ai_hub.proxy.gen_ai_hub_proxy import GenAIHubProxyClient
from ai_core_sdk.ai_core_v2_client import AICoreV2Client
from langchain.prompts import PromptTemplate
from langchain_hana import HanaDB
from pydantic import BaseModel, Field
from gen_ai_hub.proxy.langchain.init_models import init_llm, init_embedding_model

#define the local testing variable (True to skip authorization)   
local_testing = False

# Load HANA Cloud connection details
with open(os.path.join(os.getcwd(), 'env_cloud.json')) as f:
    hana_env_c = json.load(f)

# Load AI Core configuration
with open(os.path.join(os.getcwd(), 'env_config.json')) as f:
    aicore_config = json.load(f)

# Initialize the AI Core client
ai_core_client = AICoreV2Client(base_url=aicore_config['AICORE_BASE_URL'],
                            auth_url=aicore_config['AICORE_AUTH_URL'],
                            client_id=aicore_config['AICORE_CLIENT_ID'],
                            client_secret=aicore_config['AICORE_CLIENT_SECRET'],
                            resource_group=aicore_config['AICORE_RESOURCE_GROUP'])
    
# Initialize the GenAIHub proxy client        
proxy_client = GenAIHubProxyClient(ai_core_client = ai_core_client)
# Init the OpenAI embedding model
embedding_model = init_embedding_model('text-embedding-3-large', proxy_client=proxy_client)
# Set up the Chat LLM model
llm = init_llm('gpt-5', proxy_client=proxy_client, max_tokens=30000)

# Define the schema as a Pydantic model
class ProductData(BaseModel):
    category: str = Field(description="The category of the product")
    currency: str = Field(description="The currency")
    description: str = Field(description="The description of the product")
    leadTimeDays: int = Field(description="The lead time (in days)")
    manufacturer: str = Field(description="The manufacturer of the product")
    minOrder: int = Field(description="Minimal order amount for the product")
    productId: str = Field(description="The ID of the product")
    productName: str = Field(description="The name of the product")
    rating: int = Field(description="The rating of the product")
    status: str = Field(description="The status of the product")
    stockQuantity: int = Field(description="The stock quantity of the product")
    supplierAddress: str = Field(description="The address of the supplier")
    supplierCity: str = Field(description="The city where the supplier is located")
    supplierCountry: str = Field(description="The country where the supplier is located")
    supplierId: str = Field(description="The ID of the supplier")
    supplierName: str = Field(description="The name of the supplier")
    unitPrice: float = Field(description="The unit price of the product")

# --- Enable structured output on this LLM ---
llm_structured = llm.with_structured_output(
    method="json_schema",   # JSON Schema validation mode
    schema=ProductData,     # Pydantic model
    strict=True             # Strict schema validation
)
# Create a Flask application
app = Flask(__name__)
# Create an environment object to access the UAA service
# This is used for authorization
env = AppEnv()

# This function is called when the /retrieveData endpoint is hit
#The function takes the incoming data as input and returns the structured LLM output
def process_data(data, conn_db_api):
    try:
        # Create a LangChain VectorStore interface for the HANA database and specify the table (collection) to use for accessing the vector embeddings
        db_openai_table = HanaDB(
            embedding=embedding_model, 
            connection=conn_db_api, 
            table_name="PRODUCTS_IT_ACCESSORY_OPENAI_"+ hana_env_c['user'],
            content_column="VEC_TEXT", # the original text description of the product details
            metadata_column="VEC_META", # metadata associated with the product details
            vector_column="VEC_VECTOR" # the vector representation of each product 
        )
        question = data["query"]

        retriever = db_openai_table.as_retriever(search_kwargs={'k':25})
        docs = retriever.invoke(question)

        # Combine retrieved text manually
        context = "\n\n".join([doc.page_content for doc in docs])

        # Ask structured model to extract ProductData
        PRODUCT_PROMPT = PromptTemplate.from_template("""
        You are a product data extraction assistant.
        Use the context below to find detailed product information and fill in the fields of the provided schema.
        Return only information that can be directly inferred from the context.
        If a field is missing, use a reasonable default (empty string, 0, or 0.0).

        Context:
        {context}

        Question:
        {question}
        """)

        prompt_text = PRODUCT_PROMPT.format(
            context=context,
            question=question
        )

        print("Prompt Text:", prompt_text)
        
        # Use the structured LLM directly here
        response = llm_structured.invoke(prompt_text)

        print("Structured Response:", response)

        return response.model_dump_json()
        
    except Exception as e:
        return json.dumps({"error": str(e)})

if not local_testing:
    uaa_service = env.get_service(name='product_search_rag-python-srv-uaa').credentials

# Authorization Decorator
def require_auth(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not local_testing:
            if 'authorization' not in request.headers:
                return jsonify({"error": "You are not authorized to access this resource"}), 403
            
            access_token = request.headers.get('authorization')[7:]
            security_context = xssec.create_security_context(access_token, uaa_service)
            is_authorized = security_context.check_scope('uaa.resource')

            if not is_authorized:
                return jsonify({"error": "You are not authorized to access this resource"}), 403

        return f(*args, **kwargs)  # Call the original function if authorized

    return decorated_function

@app.route("/retrieveData", methods=["POST"])
@require_auth
def process_request():
    try:
        # Establish a connection to the HANA Cloud database
        conn_db_api = dbapi.connect( 
            address=hana_env_c['url'],
            port=hana_env_c['port'], 
            user=hana_env_c['user'], 
            password=hana_env_c['pwd']   
        )
        data = request.get_json()
        required_fields = ["query"]
        if not data or any(field not in data for field in required_fields):
            return jsonify({"error": "Invalid JSON input"}), 400
        
        result = process_data(data, conn_db_api)
        if len(result) < 1:
            return jsonify({"error": "no suggestions could be retrieved"}), 400
        else:
            return result, 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Always close connection after request
        if conn_db_api:
            conn_db_api.close()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))

