# this is a Flask application that uses the SAP AI Core and AI Foundation services to process queries with or without retrieval-augmented generation (RAG) capabilities.

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
from langchain.chains import RetrievalQA
from langchain_hana import HanaDB
from langchain.schema import HumanMessage

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
embedding_model = OpenAIEmbeddings(proxy_model_name='text-embedding-3-large', proxy_client=proxy_client)

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

        # Extract the required fields from the incoming data
        chat_model_name = data["chatModelName"]
        question = data["query"]
        with_RAG = data["withRAG"]
        inc_prompt = data["prompt"]

        # Set up the ChatOpenAI model
        llm = ChatOpenAI(proxy_model_name=chat_model_name, proxy_client=proxy_client)

        # If with_RAG is True, set up the RetrievalQA chain
        if with_RAG:

            # Set up the prompt template for the RetrievalQA chain
            prompt_template = f"{inc_prompt}" + """

                {context}

                question: {question}

                """

            PROMPT = PromptTemplate(template = prompt_template, 
                            input_variables=["context", "question"]
                        )
                
            chain_type_kwargs = {"prompt": PROMPT}

            # Set the top_k value for the retriever
            # This value is passed in the incoming data and specifies how many top results to retrieve from the database
            top_k = data["topK"]
            # Set up the retriever with the specified top_k value
            retriever = db_ada_table.as_retriever(search_kwargs={'k':top_k})

            # Set up the RetrievalQA chain with the specified retriever and chain type
            # The chain type is set to "stuff", which means that the retrieved documents will be passed directly to the LLM
            # The chain_type_kwargs parameter is used to pass the prompt template to the chain
            # The retriever is used to retrieve the top_k documents from the database based on the input question
            # The LLM is used to generate the final answer based on the retrieved documents and the input question
            # The answer is then returned as a JSON object
            qa = RetrievalQA.from_chain_type(llm=llm,
                            retriever=retriever, 
                            chain_type="stuff",
                            chain_type_kwargs= chain_type_kwargs)

            # Invoke the RetrievalQA chain with the input question
            answer = qa.invoke(question)

            return json.dumps(answer)
        # Otherwise, use the ChatOpenAI model directly
        else:
            # Set up the prompt template for the ChatOpenAI model
            prompt_template = f"{inc_prompt}" + """

                question: {question}

                """

            PROMPT = PromptTemplate(template = prompt_template, 
                        input_variables=["question"]
                       )

            # Invoke the ChatOpenAI model with the input question
            answer = llm.invoke(
                [HumanMessage(content=PROMPT.format(question=question))],
            )

            # Create a dictionary to hold the response
            # The dictionary contains the input question and the generated answer
            response_dict = {
                "query": question,
                "result": answer.content
            }
            return json.dumps(response_dict, indent=4)

    except Exception as e:
        return json.dumps({"error": str(e)})

# Close the database connection when the application exits
# This function is registered with the atexit module, which ensures that it is called when the application exits
# This is important to ensure that the database connection is properly closed and resources are released
def close_db_connection():
    if conn_db_api:
        conn_db_api.close()
        print("DB connection closed")
# Register the close_db_connection function with atexit
atexit.register(close_db_connection)

# Set the port for the Flask application to listen on
# The port is set to the value specified in the PORT environment variable, or 3000 if not specified
port = int(os.environ.get('PORT', 3000))
if not local_testing:
    # If not in local testing mode, get the UAA service credentials
    uaa_service = env.get_service(name='product_search_rag_YOUR_NUMBER-python-srv-uaa').credentials

# Authorization Decorator
def require_auth(f):
    @functools.wraps(f) # Preserve the original function's name and docstring
    # This decorator is used to check if the user is authorized to access the endpoint
    def decorated_function(*args, **kwargs):
        if not local_testing:
            if 'authorization' not in request.headers:
                return jsonify({"error": "You are not authorized to access this resource"}), 403
            
            # Extract the access token from the request headers
            access_token = request.headers.get('authorization')[7:]
            # Create a security context using the access token and UAA service credentials
            # The security context is used to check if the user has the required scope to access the resource
            security_context = xssec.create_security_context(access_token, uaa_service)
            # Check if the user has the required scope to access the resource
            is_authorized = security_context.check_scope('uaa.resource')

            if not is_authorized:
                return jsonify({"error": "You are not authorized to access this resource"}), 403

        return f(*args, **kwargs)  # Call the original function if authorized

    return decorated_function

# Define the /retrieveData endpoint
@app.route("/retrieveData", methods=["POST"])
@require_auth # Apply the authorization decorator to the endpoint

def process_request():
    try:
        data = request.get_json()
        required_fields = ["prompt", "query", "chatModelName", "topK", "withRAG"]
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

