from huggingface_hub import login
import os
login(os.environ["HF_TOKEN"])
import litellm
litellm.drop_params = True
from google.adk.agents import LlmAgent, SequentialAgent, LoopAgent
from google.adk.tools import ToolContext
from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from datetime import datetime
from google import genai
from google.genai import types
from google.adk.events import Event
from google.adk.sessions import InMemorySessionService, Session
from google.genai.types import Content, UserContent, Part
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
from google.adk.tools import agent_tool
from google.adk.tools import google_search
import pandas as pd
from flask import Flask, request, jsonify, Blueprint
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import numpy as np
from google.adk.models.lite_llm import LiteLlm
from rag_news import rag_news
from rag_index import rag_index
LLM_MODEL = "huggingface/together/Qwen/Qwen2.5-7B-Instruct"
SEARCH_GOOGLE_MODEL = "gemini-2.5-flash-preview-05-20"
search_agent = Agent(
    model=SEARCH_GOOGLE_MODEL,
    name='SearchAgent',
    instruction=""",
    Trả lời câu hỏi của người dùng trực tiếp bằng công cụ tìm kiếm Google; Cung cấp câu trả lời ngắn gọn và súc tích.
    Không nhất thiết phải tìm kiếm toàn bộ câu hỏi của người dùng, chỉ cần tìm kiếm phần cần sử dụng Google.
    Thay vì câu trả lời chi tiết, hãy cung cấp thông tin hành động ngay lập tức cho khách hàng, trong một câu duy nhất.
    Đừng yêu cầu người dùng tự kiểm tra hoặc tìm kiếm thông tin, đó là vai trò của bạn; hãy cố gắng cung cấp thông tin hữu ích nhất có thể.
    QUAN TRỌNG:
    - Luôn trả lời bằng dạng gạch đầu dòng
    - Chỉ rõ thông tin này quan trọng với người dùng như thế nào
    """,
    tools=[google_search],
)

search_google_agent = Agent(
    name="search_google_agent",
    model=SEARCH_GOOGLE_MODEL,
    description="Search Agent",
    tools=[agent_tool.AgentTool(agent=search_agent)],
)
# --- Constants ---
APP_NAME = "agentic_rag_finance" # New App Name
USER_ID = "dev_user_01"
SESSION_ID_BASE = "loop_exit_tool_session" # New Base Session ID
LLM = LiteLlm(model=LLM_MODEL)
# --- State Keys ---
STATE_CURRENT_DOC = "current_document"
STATE_CRITICISM = "criticism"

# Define the exact phrase the Critic should use to signal completion
COMPLETION_PHRASE = "Không tìm thấy lỗi" 

# --- Tool Definition ---
def exit_loop(tool_context: ToolContext):
  """Call this function ONLY when the critique indicates no further changes are needed, signaling the iterative process should end."""
  print(f"  [Tool Call] exit_loop triggered by {tool_context.agent_name}")
  tool_context.actions.escalate = True
  # Return empty dict as tools should typically return JSON-serializable output
  return {}

# --- Agent Definitions ---
load_dotenv()
app = Flask(__name__)
CORS(app)

index_rag = FunctionTool(rag_index)
news_rag = FunctionTool(rag_news)

index_agent = Agent(
    model=LLM,
    name="index_agent",
    description="This agent suggests key events and news given some user preferences. You can use the google_search_grounding tool to search the web for information. Be explicit and give concrete names of the events and news.",
    instruction="Bạn là một chuyên gia tài chính chuyên trả lời các câu hỏi về chỉ số tài chính, chứng khoán. Bạn sẽ nhận thông tin từ truy vấn của người dùng và sử dụng tool index_rag để trả lời câu hỏi. Giữ nội dung truy vấn càng không thay đổi càng tốt.",
    tools=[index_rag],
)
news_agent = Agent(
    model=LLM,
    name="news_agent",
    description="This agent provides warranty information for products. You can use the google_search_grounding tool to search the web for information.",
    instruction="Bạn là một chuyên gia tài chính chuyên trả lời các câu hỏi về các sự kiện tài chính, các doanh nghiệp, công ty. Bạn sẽ nhận thông tin từ truy vấn của người dùng và sử dụng tool news_rag để trả lời câu hỏi. Giữ nội dung truy vấn càng không thay đổi càng tốt.",
    tools=[news_rag],
)

manager_agent = Agent(
    model=LLM,
    name="manager_agent",
    description="This agent is responsible for managing the conversation and the tools used by the other agents.",
    instruction="""
Bạn là người quản lý các agent chuyên biệt. Vai trò của bạn là:

1. Phân tích yêu cầu của người dùng và xác định agent chuyên biệt nào có thể xử lý tốt nhất
2. Phân công nhiệm vụ cho agent phù hợp (index_agent hoặc news_agent hoặc google search)
3. Xử lý thông tin được trả về từ các agent này
4. Tổng hợp một phản hồi toàn diện cuối cùng bằng cách sử dụng dữ liệu đã thu thập

CÁC AGENT CÓ SẴN:
- index_agent: Sử dụng cho các câu hỏi về chỉ số tài chính, chứng khoán
- news_agent: Sử dụng cho các câu hỏi về các sự kiện tài chính, các doanh nghiệp, công ty
- google_search: nếu câu hỏi của người dùng không liên quan đến sản phẩm hoặc bảo hành, hãy sử dụng agent google_search để tìm kiếm thông tin trên internet
QUY TRÌNH:
1. Khi nhận được truy vấn từ người dùng, phân tích để xác định cần agent nào
2. Chuyển truy vấn cho agent đã chọn bằng cách gọi họ
3. Khi quyền điều khiển trở lại với bạn, phản hồi của agent sẽ có sẵn trong ngữ cảnh cuộc trò chuyện
4. Trích xuất thông tin liên quan từ phản hồi của agent
5. Định dạng và trình bày thông tin này trong phản hồi cuối cùng của bạn cho người dùng

Luôn ghi nhận nguồn thông tin (agent nào cung cấp) trong quá trình xử lý nội bộ, nhưng trình bày câu trả lời cuối cùng như một phản hồi thống nhất cho người dùng.
""",
    sub_agents=[index_agent, news_agent, search_google_agent],
    output_key=STATE_CURRENT_DOC
)

# STEP 1: Initial Writer Agent (Runs ONCE at the beginning)
initial_writer_agent = manager_agent

# STEP 2a: Critic Agent (Inside the Refinement Loop)
critic_agent_in_loop = LlmAgent(
    name="CriticAgent",
    model=LLM,
    include_contents='none',
    # MODIFIED Instruction: More nuanced completion criteria, look for clear improvement paths.
    instruction=f"""You are a Constructive Critic AI reviewing a document draft (typically 2-6 sentences). Your goal is balanced feedback.

    **Document to Review:**
    ```
    {{current_document}}
    ```

    **Task:**
    Review the document for clarity, engagement, and basic coherence according to the initial topic (if known).

    IF you identify 1-2 *clear and actionable* ways the document could be improved to better capture the topic or enhance reader engagement (e.g., "Needs a stronger opening sentence", "Clarify the character's goal"):
    Provide these specific suggestions concisely. Output *only* the critique text.

    ELSE IF the document is coherent, addresses the topic adequately for its length, and has no glaring errors or obvious omissions:
    Respond *exactly* with the phrase "{COMPLETION_PHRASE}" and nothing else. It doesn't need to be perfect, just functionally complete for this stage. Avoid suggesting purely subjective stylistic preferences if the core is sound.

    Do not add explanations. Output only the critique OR the exact completion phrase.
    Language: Vietnamese
""",
    description="Reviews the current draft, providing critique if clear improvements are needed, otherwise signals completion.",
    output_key=STATE_CRITICISM
)


# STEP 2b: Refiner/Exiter Agent (Inside the Refinement Loop)
refiner_agent_in_loop = LlmAgent(
    name="RefinerAgent",
    model=LLM,
    # Relies solely on state via placeholders
    include_contents='none',
    instruction=f"""You are a Creative Writing Assistant refining a document based on feedback OR exiting the process.
    **Current Document:**
    ```
    {{current_document}}
    ```
    **Critique/Suggestions:**
    {{criticism}}

    **Task:**
    Analyze the 'Critique/Suggestions'.
    IF the critique is *exactly* "{COMPLETION_PHRASE}":
    You MUST call the 'exit_loop' function. Do not output any text.
    ELSE (the critique contains actionable feedback):
    Carefully apply the suggestions to improve the 'Current Document'. Output *only* the refined document text.

    Do not add explanations. Either output the refined document OR call the exit_loop function.
""",
    description="Refines the document based on critique, or calls exit_loop if critique indicates completion.",
    tools=[exit_loop], # Provide the exit_loop tool
    output_key=STATE_CURRENT_DOC # Overwrites state['current_document'] with the refined version
)


# STEP 2: Refinement Loop Agent
refinement_loop = LoopAgent(
    name="RefinementLoop",
    # Agent order is crucial: Critique first, then Refine/Exit
    sub_agents=[
        critic_agent_in_loop,
        refiner_agent_in_loop,
    ],
    max_iterations=5 # Limit loops
)

# STEP 3: Overall Sequential Pipeline
# For ADK tools compatibility, the root agent must be named `root_agent`
root_agent = SequentialAgent(
    name="IterativeWritingPipeline",
    sub_agents=[
        initial_writer_agent, # Run first to create initial doc
        refinement_loop       # Then run the critique/refine loop
    ],
    description="Writes an initial document and then iteratively refines it with critique using an exit tool."
)

# Define agent with output_key
initial_agent = initial_writer_agent

# --- Setup Runner and Session ---
app_name, user_id, session_id = APP_NAME, USER_ID, SESSION_ID_BASE
session_service = InMemorySessionService()
runner = Runner(
    agent=initial_agent,
    app_name=app_name,
    session_service=session_service
)
session = session_service.create_session(app_name=app_name, 
                                        user_id=user_id, 
                                        session_id=session_id)
print(f"Initial state: {session.state}")

# --- Run the Agent ---
# Runner handles calling append_event, which uses the output_key
# to automatically create the state_delta.
user_message = Content(parts=[Part(text="Hello")])
for event in runner.run(user_id=user_id, 
                        session_id=session_id, 
                        new_message=user_message):
    if event.is_final_response():
      print(f"Agent responded.") # Response text is also in event.content

# --- Check Updated State ---
updated_session = session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID_BASE)
print(f"State after agent run: {updated_session.state}")