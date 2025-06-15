from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import Chroma
import pickle
from rag import query_rag
chroma_client = Chroma(persist_directory="D:/agentic_rag_finance/news_chromadb")
collection_name = "news"
all_chunked_documents = pickle.load(open("D:/agentic_rag_finance/chunked_documents_news.pkl", "rb"))
vectorstore = chroma_client.get_collection(name=collection_name)
vectorstore_retreiver = vectorstore.as_retriever(search_kwargs={"k": 10})
keyword_retriever = BM25Retriever.from_documents(all_chunked_documents)
keyword_retriever.k =  10
ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retreiver,
                                                   keyword_retriever],
                                       weights=[0.3, 0.7])
def rag_news(query_text, ensemble_retriever):
    response = query_rag(query_text, ensemble_retriever)
    return response


    