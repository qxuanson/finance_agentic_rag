import os
from huggingface_hub import InferenceClient
from datetime import datetime
client = InferenceClient(
    provider="together",
    api_key=os.environ['TOGETHER_API_KEY'],
)

def get_date_query(query_text):
  day = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages = [
        {"role": "system", "content": """Nhiệm vụ của bạn là trích xuất thông tin ngày trong câu hỏi và trả lời dưới dạng dd/mm/yyyy. Nếu không có thông tin về yyyy thì mặc định là 2025. Lưu ý các trường hợp sau đây trả về chữ none:
        Nếu không có ngày tháng rõ ràng trong câu hỏi. Nếu xuất hiện nhiều ngày hoặc khoảng thời gian trong câu hỏi."""},
        {"role": "user", "content": f"Trả lời câu hỏi này: {query_text}"},
    ]
  )
  return day.choices[0].message.content

def retrieval(query_text, ensemble_retriever):
  """
  Query a Retrieval-Augmented Generation (RAG) system using Chroma database and OpenAI.
  Args:
    - query_text (str): The text to query the RAG system with.
  Returns:
    - formatted_response (str): Formatted response including the generated text and sources.
    - response_text (str): The generated response text.
  """
  date = get_date_query(query_text)
  query_text = 'search query: ' + query_text
  response = ensemble_retriever.invoke(query_text)
  document = []
  # Sắp xếp các document theo 'publish_date' từ cũ nhất đến mới nhất
  # Chuyển đổi chuỗi ngày tháng thành đối tượng datetime để sắp xếp chính xác
  sorted_response = sorted(
      response, 
      key=lambda doc: datetime.strptime(doc.metadata['publish_date'], "%d/%m/%Y")
  )

  document = []
  # Lặp qua danh sách đã được sắp xếp
  for doc in sorted_response:
      if date == "None" or date == "none":
        document.append(doc.page_content)
      else:
        # So sánh ngày sau khi đã được sắp xếp
        if datetime.strptime(doc.metadata['publish_date'], "%d/%m/%Y").date() >= datetime.strptime(date, "%d/%m/%Y").date():
          document.append(doc.page_content)
          
  return document, date

def query_rag(query_text, ensemble_retriever):
  """
  Query a Retrieval-Augmented Generation (RAG) system using Chroma database and OpenAI.
  Args:
    - query_text (str): The text to query the RAG system with.
  Returns:
    - formatted_response (str): Formatted response including the generated text and sources.
    - response_text (str): The generated response text.
  """
  #date = get_date_query(query_text)
  #print(date)
  #print(query_text)
  document, date = retrieval(query_text, ensemble_retriever)
  docs = "\n".join(document)
  completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages = [
        {"role": "system", "content": f"""Bạn là một trợ lý AI cao cấp, chuyên tổng hợp và trình bày thông tin tài chính một cách chính xác từ các nguồn tin tức được cung cấp.
Nhiệm vụ của bạn là trả lời CÂU HỎI của người dùng. Trước khi đưa ra câu trả lời cuối cùng, hãy suy nghĩ từng bước về cách bạn sẽ sử dụng ngữ cảnh.

QUY TRÌNH BẮT BUỘC:
1.  ĐỌC KỸ CÂU HỎI: Hiểu rõ người dùng muốn biết điều gì và câu hỏi có publish_date = {date}
2.  QUÉT NGỮ CẢNH: Tìm kiếm tất cả các "Tài liệu" trong NGỮ CẢNH có chứa thông tin liên quan đến CÂU HỎI. Chú ý đến NGÀY/publish_date của mỗi tài liệu.
3.  LỰA CHỌN VÀ TỔNG HỢP:
* Ưu tiên thông tin từ các tài liệu có publish_date trùng lặp với {date} hoặc gần nhất.
* Nếu các tài liệu cung cấp các khía cạnh khác nhau của vấn đề, hãy tổng hợp chúng lại.
* Trích xuất chính xác các số liệu tài chính.
4.  TRƯỜNG HỢP KHÔNG CÓ THÔNG TIN: Nếu sau khi quét kỹ NGỮ CẢNH mà không tìm thấy thông tin liên quan, hãy trả lời: "Dựa trên các tài liệu được cung cấp, tôi không có thông tin nào về [chủ đề câu hỏi]."
5.  CẤU TRÚC CÂU TRẢ LỜI: Trình bày câu trả lời một cách rõ ràng, mạch lạc. Nếu câu hỏi có nhiều phần, hãy cố gắng trả lời từng phần.

NGỮ CẢNH:
--- BẮT ĐẦU NGỮ CẢNH ---
[context]
--- KẾT THÚC NGỮ CẢNH ---

CÂU HỎI:
[question]

SUY NGHĨ TỪNG BƯỚC (bạn không cần hiển thị phần này trong câu trả lời cuối cùng, nhưng hãy tuân theo quy trình này):
1.  Câu hỏi đang hỏi về: ...
2.  Các tài liệu liên quan trong ngữ cảnh: [Liệt kê tài liệu, nguồn, ngày]...
3.  Thông tin chính từ mỗi tài liệu: ...
4.  Lựa chọn thông tin/số liệu và nguồn/ngày tương ứng để trả lời: ...

TRẢ LỜI CUỐI CÙNG (tuân thủ các quy tắc): chỉ trả về câu trả lời cuối cùng, không trả về quá trình suy nghĩ.
        """},
        {"role": "user", "content": f"Trả lời câu hỏi này: {query_text}, dựa trên tài liệu sau: {docs}"},
    ],
    )
  return completion.choices[0].message.content