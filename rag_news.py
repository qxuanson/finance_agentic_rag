from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import Chroma
import pickle
from rag import query_rag
from database_news import embeddings

def rag_index(query_text):
    vectorstore = Chroma(persist_directory="/news_chromadb", collection_name = "news", embedding_function=embeddings)
    all_chunked_documents = pickle.load(open("/chunked_documents_index.pkl", "rb"))
    vectorstore_retreiver = vectorstore.as_retriever(search_kwargs={"k": 10})
    keyword_retriever = BM25Retriever.from_documents(all_chunked_documents)
    keyword_retriever.k =  10
    ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retreiver,
                                                    keyword_retriever],
                                        weights=[0.3, 0.7])
    response = query_rag(query_text, ensemble_retriever)
    return response
