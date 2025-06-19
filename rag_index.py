from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import Chroma
import pickle
from rag import query_rag
#import chromadb
from embedding import get_embedding
from embedding import embeddings
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
model_kwargs = {'trust_remote_code': True}
embedding = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5",
                                       model_kwargs=model_kwargs)

collection_name = "index"
persist_directory = "./index_chromadb"
vectorstore = Chroma(
        collection_name=collection_name, 
        embedding_function=embedding,
        persist_directory=persist_directory
    )

def rag_index(query: str) -> str:
    all_chunked_documents = pickle.load(open("./chunked_documents_index.pkl", "rb"))
    vectorstore_retreiver = vectorstore.as_retriever(search_kwargs={"k": 10})
    keyword_retriever = BM25Retriever.from_documents(all_chunked_documents)
    keyword_retriever.k =  10
    ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retreiver,
                                                      keyword_retriever],
                                          weights=[0.3, 0.7])
    response = query_rag(query, ensemble_retriever)
    return response