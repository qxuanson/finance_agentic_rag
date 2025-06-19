from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
#from langchain_huggingface.embeddings import HuggingFaceEmbeddings
app = Flask(__name__)

local_model_path = "./nomic-embed-text-v1.5-local"
embeddings = SentenceTransformer(local_model_path, trust_remote_code=True)

@app.route('/get_embeddings', methods=['POST'])
def get_embedding(text: str) -> list[float]:
    response = embeddings.encode(text)
    return response
