from flask import Flask, request, jsonify
import os
import boto3
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema
from pymilvus import DataType, utility
from dotenv import load_dotenv
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize S3 client
s3_client = boto3.client("s3", region_name='us-east-1')

# Initialize SentenceTransformer model with BERT
model = SentenceTransformer('sentence-transformers/all-MPNet-base-v2')

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")  # Default to localhost if no env var set
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")  # Default port

# Try to get the username and password from environment variables (for remote)
milvus_user = os.getenv("MILVUS_USER")
milvus_password = os.getenv("MILVUS_PASSWORD")

# Connect to Milvus, conditionally adding username and password if they exist
if milvus_user and milvus_password:
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        user=milvus_user,
        password=milvus_password
    )
    print("✅ Connected to Milvus with user authentication")
else:
    # If no user and password are provided (local setup), connect without them
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )
    print("✅ Connected to Milvus (local)")

def create_or_get_collection(parent_collection_name, subfolder_name):
    """
    Ensure the Milvus collection and sub-collection exist.
    """
    full_collection_name = f"{parent_collection_name}_{subfolder_name}"

    if utility.has_collection(full_collection_name):
        collection = Collection(full_collection_name)
        print(f"==>Collection '{full_collection_name}' exists with {collection.num_entities} entities")
        return collection

    # Define schema for the collection
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),  # Adjust 'dim' based on your embedding model
    ]
    schema = CollectionSchema(fields=fields, description="Text embeddings collection")
    collection = Collection(name=full_collection_name, schema=schema)
    print(f"✅ Collection '{full_collection_name}' created")
    return collection

@app.route('/process-file', methods=['POST'])
def process_file():
    data = request.json
    bucket_name = data.get('bucket_name')
    file_key = data.get('file_key')

    if not bucket_name or not file_key:
        return jsonify({"error": "Missing bucket_name or file_key"}), 400

    print(f"Processing file from bucket '{bucket_name}' with key '{file_key}'....")

    # Download the file from S3
    file_obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    file_content = file_obj['Body'].read().decode('utf-8')
    print("✅ Downloaded file from S3")

    # Generate vector
    vector = model.encode(file_content)
    print("✅ Vectorized data")

    # Ensure the vector is a list of lists before inserting
    vector = [vector.tolist()] if isinstance(vector, np.ndarray) else [vector]

    # Parse folder and subfolder from file_key
    path_parts = file_key.split('/')
    if len(path_parts) < 3:  # Ensure it has at least folder/subfolder/file
        return jsonify({"error": f"Invalid file path: {file_key}"}), 400

    parent_folder = path_parts[0]
    subfolder = path_parts[1]

    # Insert into Milvus    
    collection = create_or_get_collection(parent_folder, subfolder)
    print(f"==>Number of entities before insertion: {collection.num_entities}")

    try:
        collection.insert([vector])
        collection.flush()
        print("✅ Inserted data into Milvus")
        print(f"==>Number of entities after insertion: {collection.num_entities}")
        return jsonify({"message": f"Successfully inserted vector for {file_key}"}), 200
    except Exception as e:
        print(f"(Error inserting vector: {str(e)})")
        return jsonify({"error": f"Error inserting vector: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
