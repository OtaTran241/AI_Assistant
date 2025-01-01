from typing import Callable, Any, Dict, List
import os
import json
import re
import threading
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from Agentic_Rag.Agent_tools import tools
from dotenv import load_dotenv

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
faiss_db = FAISS.load_local("datas/Rag_datas/Vector_db", embedding_model, allow_dangerous_deserialization=True)

os.environ["GOOGLE_API_KEY"] = os.getenv("Google_API_Key")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.6, max_output_tokens=512)

chat_history: List[Dict[str, str]] = []

content_summary_prompt_template = """
    Dựa trên các tài liệu sau đây, hãy tổng hợp thông tin quan trọng liên quan đến câu hỏi: {question}
    Chỉ cần lấy các nội dung liên quan đến câu hỏi từ các tài liệu, không thêm thông tin bên ngoài.
    Nếu không có thông tin liên quan đến câu hỏi trong các tài liệu thì trả về không có thông tin.

    Các tài liệu:
    {documents}

    các thông tin liên quan đến câu hỏi:
"""

history_summary_prompt_template = """
    Dựa trên lịch sử cuộc trò chuyện sau, hãy tóm tắt những ý chính của cả user lẫn assistant.
    Chỉ cần tóm tắt các nội dung quan trọng, không thêm thông tin ngoài những gì được cung cấp.

    Lịch sử cuộc trò chuyện:
    {history}

    Tóm tắt:
"""

tool_usage_prompt_template = """
    Bạn là một Agent thông minh. Nếu câu hỏi yêu cầu thông tin có thể lấy từ các tool dưới đây, hãy sử dụng tool tương ứng.

    Danh sách tools:
    {tools}

    Khi trả lời, chỉ cần trả về : {{"tool_name": "<tên_tool>", "arguments": {{<các arguments>}}}}
    Nếu không sử dụng tool, trả về: {{}}

    Ví dụ:
    - "Thời tiết Hồ Chí Minh thế nào?" -> {{"tool_name": "get_weather", "arguments": {{"location": "Hồ Chí Minh"}}}}
    - "Tính diện tích với cao 20 và dài 30" -> {{"tool_name": "calculate_area", "arguments": {{"width": 30, "height": 20}}}}
    - "Bạn khỏe không?" -> {{}}

    Câu hỏi: {question}
    Trả lời:
"""

qa_system_prompt_template = """
    Bạn là một trợ lý thông minh tên là Toka. Hãy trả lời câu hỏi sau, khi trả lời hãy làm theo thứ tự ưu tiên sau:

    1. Nếu có thông tin từ công cụ hãy sử dụng.
    2. Nếu không, hãy sử dụng thông tin từ dữ liệu.
    3. Sử dụng thông tin từ lịch sử trò chuyện nếu cần.
    3. Nếu không có thông tin nào, hãy trả lời một cách chính xác và rõ ràng.

    Câu hỏi: {question}

    Thông tin từ công cụ: {tool_response}
    Thông tin từ dữ liệu: {context}
    Thông tin lịch sử trò chuyện: {chat_history}

    Câu trả lời:
"""

def summarize_history_thread(chat_history, result_holder):
    result_holder["history_summary"] = agent_summarize_chat_history(chat_history)

def retrieve_context_thread(user_input, result_holder):
    result_holder["context"] = agent_retriever(user_input)

def use_tool_thread(user_input, result_holder):
    result_holder["tool_response"] = agent_uses_tool(user_input)

def agent_summarize_chat_history(history: List[Dict[str, str]]) -> str:
    formatted_history = "\n".join(
        [f"Người dùng: {entry['user']}\nToka: {entry['assistant']}" for entry in history]
    )
    prompt = PromptTemplate(
        input_variables=["history"],
        template=history_summary_prompt_template
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    summary = chain.run(history=formatted_history)
    return summary.strip()

def agent_retriever(question: str) -> str:
    if not faiss_db:
        return ""

    retriever = faiss_db.as_retriever(search_kwargs={"k": 5})
    context_documents = retriever.get_relevant_documents(question)

    aggregated_content = "\n".join([f"- {doc.page_content}" for doc in context_documents])

    summary_prompt = PromptTemplate(
        input_variables=["documents"],
        template=content_summary_prompt_template
    )

    chain = LLMChain(llm=llm, prompt=summary_prompt)
    summary = chain.run(documents=aggregated_content, question=question)

    return summary.strip()

def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    for tool in tools:
        if tool.name == tool_name:
            return tool.call(**arguments)
    return f"Tool '{tool_name}' không tồn tại."

def agent_uses_tool(question: str) -> str:
    tool_list_prompt = "\n".join([tool.get_header() for tool in tools])
    prompt = PromptTemplate(
        input_variables=["tools", "question"],
        template=tool_usage_prompt_template
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    tool_usage_response = chain.run(tools=tool_list_prompt, question=question)
    tool_usage_response = tool_usage_response.strip("```json").strip()

    match = re.search(r'{.*}', tool_usage_response, re.DOTALL)
    if match:
        valid_json = match.group(0)
        try:
            data = json.loads(valid_json)
        except json.JSONDecodeError:
            return "Lỗi khi parse JSON"
    else:
        return "Không tìm thấy JSON hợp lệ trong câu trả lời của agent tool."

    if "tool_name" in data and "arguments" in data:
        tool_name = data["tool_name"]
        arguments = data["arguments"]
        return execute_tool(tool_name, arguments)

    return "Không sử dụng tool hoặc không thể xử lý yêu cầu."

def generate_final_response(question: str, context: str, history_summary: str, tool_response: str) -> str:
    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "tool_response", "question"],
        template=qa_system_prompt_template
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(context=context, chat_history=history_summary, tool_response=tool_response, question=question)
    return response.strip()

def main_workflow(user_input: str) -> str:
    result_holder = {
        "history_summary": None,
        "context": None,
        "tool_response": None
    }

    history_thread = threading.Thread(target=summarize_history_thread, args=(chat_history, result_holder))
    context_thread = threading.Thread(target=retrieve_context_thread, args=(user_input, result_holder))
    tool_thread = threading.Thread(target=use_tool_thread, args=(user_input, result_holder))

    history_thread.start()
    context_thread.start()
    tool_thread.start()

    history_thread.join()
    context_thread.join()
    tool_thread.join()

    history_summary = result_holder["history_summary"]
    context = result_holder["context"]
    tool_response = result_holder["tool_response"]

    final_response = generate_final_response(user_input, context, history_summary, tool_response)

    chat_history.append({"user": user_input, "assistant": final_response})
    return final_response

def get_response(user_input: str) -> str:
    return main_workflow(user_input)

if __name__ == "__main__":
    print("Chào mừng bạn đến với trợ lý thông minh Toka!")
    while True:
        user_query = input("Nhập câu hỏi của bạn (hoặc 'thoát' để dừng): ")
        if user_query.lower() == "thoát":
            print("Chào tạm biệt!")
            break
        response = get_response(user_query)
        print(f"Toka: {response}")
