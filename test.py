# docrag-mac
# Microservices:
# 1. Embedding Service (Python): Handles PDF processing, chunking, embedding generation, and storage.
# 2. Query Service (Python): Handles user queries, similarity search, context retrieval, and LLM interaction.
# 3. UI Service (Python - Flask):  Provides a web interface for user interaction.
# 4. Embedding Model Server (Shell Script + Python/other, depending on chosen LLM framework): Serves the embedding LLM.
# 5. Response Model Server (Shell Script + Python/other, depending on chosen LLM framework): Serves the response LLM.
# 6. ChromaDB Server (Shell Script):  Manages the ChromaDB vector database.
# 7. MongoDB Server (Shell Script): Manages the MongoDB document store.

import os
import json
import subprocess

def create_directory_structure():
    """Creates the necessary directory structure."""
    directories = [
        "data/pdf_documents",
        "data/db/chromadb",
        "data/db/mongodb",
        "services/embedding",
        "services/query",
        "services/ui",
        "services/embedding_model",
        "services/response_model",
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def create_config_file():
    """Creates the config.json file with default values."""
    config = {
        "pdf_documents_path": "data/pdf_documents",
        "chromadb_path": "data/db/chromadb",
        "mongodb_url": "mongodb://localhost:27017/",  # Default MongoDB URL
        "mongodb_database_name": "document_store",
        "mongodb_collection_name": "documents",
        "embedding_model_server_url": "http://localhost:8000", #Example embedding model server URL
        "response_model_server_url": "http://localhost:8001",  # Example response model server URL
        "embedding_model_startup_command": "python services/embedding_model/server.py", # Example command
        "response_model_startup_command": "python services/response_model/server.py",   # Example command
        "chromadb_startup_command": "chroma run --path data/db/chromadb",  #Example
        "mongodb_startup_command": "mongod --dbpath data/db/mongodb",
        "chunk_size": 100,
        "chunk_overlap": 20,
        "similarity_search_N": 5, # Top N similar documents
        "similarity_search_K": 3,  # Top K after filtering
        "embedding_model_name": "all-MiniLM-L6-v2", # Example, user should replace
        "response_model_name": "google/flan-t5-small"   # Example, user should replace
    }

    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)

# --- Embedding Service --- (services/embedding/embedding_service.py)
def create_embedding_service():
    """Creates the embedding service script."""
    with open("services/embedding/embedding_service.py", "w") as f:
        f.write("""
import os
import json
import chromadb
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from pymongo import MongoClient

def embed_documents():
    # Load configuration
    with open("../../config.json", "r") as f:
        config = json.load(f)

    pdf_path = config["pdf_documents_path"]
    chromadb_path = config["chromadb_path"]
    mongodb_url = config["mongodb_url"]
    db_name = config["mongodb_database_name"]
    collection_name = config["mongodb_collection_name"]
    chunk_size = config["chunk_size"]
    chunk_overlap = config["chunk_overlap"]
    embedding_model_name = config["embedding_model_name"]

    # Initialize MongoDB client
    client = MongoClient(mongodb_url)
    db = client[db_name]
    collection = db[collection_name]
    collection.drop() # Start fresh each time (for this example)

    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path=chromadb_path)


    # Load embedding model.  Use sentence-transformers (local) for simpler setup.
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    for filename in os.listdir(pdf_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_path, filename)
            print(f"Processing {filename}...")

            loader = PyPDFLoader(filepath)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            texts = text_splitter.split_documents(documents)

            # Create and store embeddings in ChromaDB
            db = Chroma.from_documents(texts, embeddings, client=chroma_client, collection_name="docrag_collection")

            # Store document chunks in MongoDB
            for i, doc in enumerate(texts):
                doc_metadata = doc.metadata
                doc_metadata['page_content'] = doc.page_content # Store text content
                doc_metadata['chunk_id'] = i  # Add a chunk ID
                doc_metadata['source_file'] = filename
                collection.insert_one(doc_metadata)
    print("Embedding complete.")

if __name__ == "__main__":
    embed_documents()
""")

# --- Query Service --- (services/query/query_service.py)
def create_query_service():
    """Creates the query service script."""
    with open("services/query/query_service.py", "w") as f:
        f.write("""
import json
import chromadb
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from pymongo import MongoClient
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def answer_query(query):
    # Load configuration
    with open("../../config.json", "r") as f:
        config = json.load(f)

    chromadb_path = config["chromadb_path"]
    mongodb_url = config["mongodb_url"]
    db_name = config["mongodb_database_name"]
    collection_name = config["mongodb_collection_name"]
    embedding_model_name = config["embedding_model_name"]
    response_model_name = config["response_model_name"]
    similarity_search_n = config["similarity_search_N"]
    similarity_search_k = config["similarity_search_K"]

    # Initialize MongoDB client
    client = MongoClient(mongodb_url)
    db = client[db_name]
    collection = db[collection_name]

   # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path=chromadb_path)

    # Load embedding model
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Load vectorstore
    vectordb = Chroma(client=chroma_client, collection_name="docrag_collection", embedding_function=embeddings)

    # Retrieve documents
    retriever = vectordb.as_retriever(search_kwargs={"k": similarity_search_n})
    docs = retriever.get_relevant_documents(query)

    # Further filter documents (if needed) - example filtering by score:
    filtered_docs = sorted(docs, key=lambda doc: doc.metadata.get('score', 0), reverse=True)[:similarity_search_k]

    # Load response LLM (using Langchain's HuggingFaceHub for simplicity)
    llm = HuggingFaceHub(repo_id=response_model_name, model_kwargs={"temperature":0.2, "max_length":512})

    # Prompt Engineering
    template = """You are a helpful assistant that answers questions based on the provided context.
    If the answer cannot be found in the context, respond with "I don't know."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": prompt}

    # RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False, chain_type_kwargs=chain_type_kwargs)
    result = qa_chain({"query": query})

    return result['result']

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    answer = answer_query(user_query)
    print(f"Answer: {answer}")
""")

# --- UI Service --- (services/ui/ui_service.py)
def create_ui_service():
    with open("services/ui/ui_service.py", "w") as f:
        f.write("""
from flask import Flask, render_template, request, jsonify
import sys
sys.path.append('../query') # Add query service directory to path
from query_service import answer_query

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_query = request.form['query']
    answer = answer_query(user_query)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Run on a different port than the LLM servers
""")
    with open("services/ui/templates/index.html", "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Document Query</title>
    <style>
        body { font-family: sans-serif; }
        .container { width: 80%; margin: 0 auto; }
        textarea { width: 100%; height: 100px; }
        button { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
        #answer { margin-top: 20px; padding: 10px; border: 1px solid #ddd; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Document Query</h1>
        <form id="queryForm">
            <textarea name="query" placeholder="Enter your query here..."></textarea><br><br>
            <button type="submit">Submit Query</button>
        </form>
        <div id="answer"></div>
    </div>

    <script>
        document.getElementById('queryForm').addEventListener('submit', function(e) {
            e.preventDefault();
            var formData = new FormData(this);
            fetch('/query', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('answer').innerText = data.answer;
            });
        });
    </script>
</body>
</html>
""")
# --- Embedding Model Server --- (services/embedding_model/server.py)
# This is a placeholder. A real implementation would depend on the chosen LLM framework.
# This example assumes a simple Hugging Face Transformers setup.
def create_embedding_model_server():
     with open("services/embedding_model/server.py", "w") as f:
        f.write("""
# Placeholder for embedding model server.
# Requires a proper LLM server setup (e.g., using Flask, FastAPI, or a dedicated framework).
# This is a very basic example and needs substantial modification for a real-world deployment.

from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)

# Load model and tokenizer (replace with your model)
tokenizer = AutoTokenizer.from_pretrained("all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("all-MiniLM-L6-v2")


@app.route('/embed', methods=['POST'])
def embed():
    data = request.get_json()
    text = data['text']
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).tolist()  # Example: mean pooling
    return jsonify({'embeddings': embeddings})

if __name__ == '__main__':
    app.run(debug=True, port=8000)  # Example port
""")

# --- Response Model Server --- (services/response_model/server.py)
# Similar to the embedding server, this is a placeholder and depends on the chosen framework.
def create_response_model_server():
    with open("services/response_model/server.py", "w") as f:
        f.write("""
# Placeholder for response model server.
# Requires a proper LLM server setup.

from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# Load model and tokenizer (replace with your model)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    input_text = data['input_text']
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return jsonify({'response': decoded_output[0]}) # Return the first generated response.

if __name__ == '__main__':
    app.run(debug=True, port=8001) # Example port

""")

# --- Startup Script --- (startup.sh)
def create_startup_script():
    """Creates a startup script to start all services."""
    with open("startup.sh", "w") as f:
        f.write("""
#!/bin/bash

# Start MongoDB
echo "Starting MongoDB..."
{config[mongodb_startup_command]} &

# Start ChromaDB
echo "Starting ChromaDB..."
{config[chromadb_startup_command]} &

# Start Embedding Model Server (example - adapt to your server setup)
echo "Starting Embedding Model Server..."
{config[embedding_model_startup_command]} &

# Start Response Model Server (example - adapt to your server setup)
echo "Starting Response Model Server..."
{config[response_model_startup_command]} &


# Start Embedding Service (one-time run, then comment out)
#echo "Running Embedding Service..."
#python services/embedding/embedding_service.py

# Start UI Service
echo "Starting UI Service..."
python services/ui/ui_service.py &

echo "All services started.  UI available at http://localhost:5000"

        """.format(config=config))
    os.chmod("startup.sh", 0o755)  # Make executable

# --- Shutdown Script --- (shutdown.sh)
def create_shutdown_script():
    with open("shutdown.sh", "w") as f:
        f.write("""
#!/bin/bash

# Find and kill processes (using pkill, which is safer than killall)
# Be very careful with pkill -n; ensure the process names are unique!

echo "Shutting down services..."

pkill -f "python services/ui/ui_service.py"
pkill -f "chroma run"
pkill -f "mongod"
pkill -f "services/embedding_model/server.py"  # Adjust if your server command is different
pkill -f "services/response_model/server.py"   # Adjust if your server command is different

echo "Services shut down."
        """)
    os.chmod("shutdown.sh", 0o755)

# --- Main execution flow ---

create_directory_structure()
create_config_file() # Creates the config file
with open("config.json", "r") as f:
    config = json.load(f)
create_embedding_service()
create_query_service()
create_ui_service()
create_embedding_model_server()
create_response_model_server()
create_startup_script()
create_shutdown_script()

print("Project setup complete.  Run ./startup.sh to start the services.")
print("After the first run, and after adding PDF documents, run the embedding service:")
print("  python services/embedding/embedding_service.py")
