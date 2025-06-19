import pickle
import chromadb
import uuid
from embedding import get_embedding # Giữ lại hàm embedding của bạn
from tqdm import tqdm # Thêm thư viện tqdm để theo dõi tiến trình đẹp hơn

print("Loading chunked documents...")
# 1. Tải các document đã được chia nhỏ
# Giả định đây là một danh sách các đối tượng LangChain Document
# Ví dụ: [Document(page_content="...", metadata={...}), Document(page_content="...", metadata={...})]
try:
    with open("./chunked_documents_index.pkl", "rb") as f:
        all_chunked_documents = pickle.load(f)
except FileNotFoundError:
    print("Error: Pickle file not found. Please check the path.")
    exit()

print(f"Loaded {len(all_chunked_documents)} documents.")

# 2. Chuẩn bị dữ liệu cho ChromaDB
# Tạo các danh sách riêng biệt cho documents, metadatas, và ids một cách rõ ràng
documents_to_add = [doc.page_content for doc in all_chunked_documents]
metadatas_to_add = [doc.metadata for doc in all_chunked_documents]
ids_to_add = [str(uuid.uuid4()) for _ in documents_to_add]

# Lọc ra các document rỗng để tránh lỗi
valid_indices = [i for i, doc in enumerate(documents_to_add) if doc and isinstance(doc, str)]

if len(valid_indices) != len(documents_to_add):
    print(f"Warning: Found and removed {len(documents_to_add) - len(valid_indices)} empty or invalid documents.")
    documents_to_add = [documents_to_add[i] for i in valid_indices]
    metadatas_to_add = [metadatas_to_add[i] for i in valid_indices]
    ids_to_add = [ids_to_add[i] for i in valid_indices]

# 3. Tạo embeddings cho các document hợp lệ
print(f"\nStarting embedding process for {len(documents_to_add)} valid documents...")
embeddings_to_add = []
# Sử dụng tqdm để theo dõi tiến trình
for doc_text in tqdm(documents_to_add, desc="Embedding Documents"):
    embeddings_to_add.append(get_embedding(doc_text))
print("Embedding process completed!")

# 4. Thiết lập ChromaDB và thêm dữ liệu
print("\nSetting up ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./index_chromadb")

# Sử dụng get_or_create_collection để tránh lỗi nếu collection đã tồn tại
collection_name = "index"
collection = chroma_client.create_collection(name=collection_name)

print(f"Adding {len(documents_to_add)} documents to collection '{collection_name}'...")
# Thêm đồng thời tất cả dữ liệu vào collection
# *** THÊM THAM SỐ `documents` QUAN TRỌNG VÀO ĐÂY ***
collection.add(
    ids=ids_to_add,
    embeddings=embeddings_to_add,
    documents=documents_to_add,  # <-- Đây là phần bị thiếu
    metadatas=metadatas_to_add
)

print("\nProcess completed successfully!")
print(f"Total documents in collection: {collection.count()}")