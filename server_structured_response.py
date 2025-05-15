import os
import json
import atexit
from flask import Flask, request, jsonify
from hdbcli import dbapi
from cfenv import AppEnv
from sap import xssec
import functools
from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings, ChatOpenAI
from gen_ai_hub.proxy.gen_ai_hub_proxy import GenAIHubProxyClient
from ai_core_sdk.ai_core_v2_client import AICoreV2Client
from langchain.prompts import PromptTemplate
from langchain_hana import HanaDB
from langchain.schema import HumanMessage
from pydantic import BaseModel, Field

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
embedding_model = OpenAIEmbeddings(proxy_model_name='text-embedding-ada-002', proxy_client=proxy_client)
# Set up the ChatOpenAI model
llm = ChatOpenAI(proxy_model_name='gpt-4o', proxy_client=proxy_client)

# Establish a connection to the HANA Cloud database
conn_db_api = dbapi.connect( 
    address=hana_env_c['url'],
    port=hana_env_c['port'], 
    user=hana_env_c['user'], 
    password=hana_env_c['pwd']   
)

# Create a LangChain VectorStore interface for the HANA database and specify the table (collection) to use for accessing the vector embeddings
db_ada_table = HanaDB(
    embedding=embedding_model, 
    connection=conn_db_api, 
    table_name="PRODUCTS_IT_ACCESSORY_ADA_"+ hana_env_c['user'],
    content_column="VEC_TEXT", # the original text description of the product details
    metadata_column="VEC_META", # metadata associated with the product details
    vector_column="VEC_VECTOR" # the vector representation of each product 
)
# Create a Flask application
app = Flask(__name__)
# Create an environment object to access the UAA service
# This is used for authorization
env = AppEnv()

# This function is called when the /retrieveData endpoint is hit
#The function takes the incoming data as input and returns the result as output

def process_data(data):
    try:
        inc_prompt = data["prompt"]

        prompt_template = f"{inc_prompt}" + """

            {context}

            question: {question}

            """

        PROMPT = PromptTemplate(template = prompt_template, 
                        input_variables=["context", "question"]
                       )

        question = data["query"]
        retriever = db_ada_table.as_retriever(search_kwargs={'k':25})
        retrieved_docs = retriever.invoke(question)

        # Combine retrieved docs into a single context string
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Define the schema as a Pydantic model
        class ProductData(BaseModel):
            productId: str = Field(description="The ID of the product")
            productName: str = Field(description="The name of the product")
            category: str = Field(description="The category of the product")
            description: str = Field(description="The description of the product")
            unitPrice: float = Field(description="The unit price of the product")
            currency: str = Field(description="The currency")
            supplierId: str = Field(description="The ID of the supplier")
            supplierName: str = Field(description="The name of the supplier")
            supplierCountry: str = Field(description="The country where the supplier is located")
            supplierCity: str = Field(description="The city where the supplier is located")
            supplierAddress: str = Field(description="The address of the supplier")
            leadTimeDays: int = Field(description="The lead time (in days)")
            minOrder: int = Field(description="Minimal order amount for the product")
            rating: int = Field(description="The rating of the product")
            status: str = Field(description="The status of the product")

        # Get the function definition
        function_def = {
            "name": "get_product_data",
            "description": "Retrieve product information",
            "parameters": ProductData.schema()  # Converts Pydantic model to OpenAI function format
        }

        answer = llm.invoke(
            [HumanMessage(content=PROMPT.format(context=context, question=question))],
            functions=[function_def],  # Pass JSON schema
            function_call={"name": "get_product_data"}  # Let the model decide when to use the function
        )

        # Parse the structured output
        parsed_output = ProductData.parse_raw(answer.additional_kwargs["function_call"]["arguments"])

        return json.dumps(parsed_output.dict())
        
    except Exception as e:
        return json.dumps({"error": str(e)})


def close_db_connection():
    if conn_db_api:
        conn_db_api.close()
        print("DB connection closed")

atexit.register(close_db_connection)

port = int(os.environ.get('PORT', 3000))
if not local_testing:
    uaa_service = env.get_service(name='product_search_rag_YOUR_NUMBER-python-srv-uaa').credentials

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
        data = request.get_json()
        required_fields = ["query", "prompt"]
        if not data or any(field not in data for field in required_fields):
            return jsonify({"error": "Invalid JSON input"}), 400
        
        result = process_data(data)
        if len(result) < 1:
            return jsonify({"error": "no suggestions could be retrieved"}), 400
        else:
            response = app.response_class(
                response=result,
                status=200,
                mimetype='application/json'
            )
            return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)

