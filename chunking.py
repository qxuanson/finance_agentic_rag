import os
import pandas as pd
from huggingface_hub import login
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import pickle
login(os.environ["HF_TOKEN"])

file_path = "news.csv"
df = pd.read_csv(file_path)
def join_string(item):
    for i in range(len(item)):
        title, published_date, description, content= item

        final_string = ""
        if title:
            final_string += f"{title}."

        if published_date:
            final_string += f"Ngày đăng: {published_date}."

        if description:
            description = description.replace("<br>", " ").replace("\n", " ")
            final_string += f" {description}"

        if content:
            content = content.replace("<br>", " ").replace("\n", " ")
            final_string += f" {content}"

    return final_string

df['information'] = df[
    [
     'title',
     'publish_date',
     'description',
     'content'
    ]
    ].astype(str).apply(join_string, axis=1)
df = df[df['information'].notna()]
df.to_csv("D:/agentic_rag_finance/news_chunking.csv", index=False)
model_kwargs = {'trust_remote_code': True}
embeddings_index = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5",
                                       model_kwargs=model_kwargs)
text_splitter = SemanticChunker(
    embeddings_index,
    breakpoint_threshold_type="percentile",
)
all_chunked_documents = []
for index, row in df.iterrows():
    text_to_split = row['information']
    publish_date = row['publish_date'] 
    documents_to_split = [{"page_content": text_to_split, "metadata": {"publish_date": publish_date}}]

    chunks = text_splitter.create_documents([text_to_split])
    for i, chunk_doc in enumerate(chunks):
        chunk_doc.metadata['publish_date'] = publish_date
        all_chunked_documents.append(chunk_doc)

print(f"\nĐã xử lý xong. Tổng số chunk được tạo: {len(all_chunked_documents)}")

save_path = "D:/agentic_rag_finance/chunked_documents_news.pkl"

with open(save_path, 'wb') as f:
    pickle.dump(all_chunked_documents, f)

print(f"Saved chunked documents to {save_path}")
