from typing import Callable, Any, Dict, List
import os
import json
import re
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from RAG.Agent_tools import tools
from dotenv import load_dotenv

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

faiss_db = FAISS.load_local("datas/Rag_datas/Vector_db", embedding_model, allow_dangerous_deserialization = True)

os.environ["GOOGLE_API_KEY"] = os.getenv("Google_API_Key")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.6, max_output_tokens=512)

chat_history: List[Dict[str, str]] = []

content_summary_prompt_template = """
    Dựa trên các tài liệu sau đây, hãy tổng hợp thông tin quan trọng liên quan đến câu hỏi: {question}
    Chỉ cần lấy các nội dung liên quan từ các tài liệu, không thêm hoặc bớt thông tin được cung cấp.

    Các tài liệu:
    {documents}

    Tóm tắt thông tin liên quan:
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

qa_system_prompt_template ="""
    Bạn là một trợ lý thông minh tên là Toka. Hãy trả lời câu hỏi sau, khi trả lời hãy làm theo thứ tự ưu tiên sau:

    1. Nếu có thông tin từ công cụ hãy sử dụng.
    2. Nếu không, hãy sử dụng thông tin từ dữ liệu.
    3. Sử dụng thông tin từ lịch sử trò chuyện.
    3. Nếu không có thông tin nào, hãy trả lời một cách sáng tạo và rõ ràng.

    Câu hỏi: {question}

    Thông tin từ công cụ: {tool_response}
    Thông tin từ dữ liệu: {context}
    Thông tin lịch sử trò chuyện: {chat_history}

    Câu trả lời:
"""

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
        except json.JSONDecodeError as e:
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
    history_summary = agent_summarize_chat_history(chat_history)

    context = agent_retriever(user_input)

    tool_response = agent_uses_tool(user_input)

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
        response = main_workflow(user_query)
        print(f"Toka: {response}")












# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.prompts import MessagesPlaceholder
# from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_core.messages import HumanMessage
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import LlamaCpp

# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# model_dir = "/content/Models/llama-7b-chat_q5_0.gguf"

# embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# vector_db_path = "data"

# n_gpu_layers = -1

# n_batch = 512

# qa_system_prompt = """You are an assistant for question-answering tasks. Your name is Toka. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say you don't know, don't try to make one up. Don't make a new question\n
# {context}"""


# cq_system_prompt = """Given a chat history and the latest user question \
# which might reference context in the chat history. \
# Don't make a new question.\
# If you Don't know the answer, say you Don't know, Don't try to make one up."""

# def load_llm(model_file):
#     llm = LlamaCpp(
#         model_path=model_file,
#         n_gpu_layers=n_gpu_layers,
#         n_batch=n_batch,
#         callback_manager=callback_manager,
#         verbose=True,
#         max_tokens=1024,
#         temperature=0.01,
#         n_ctx=1024,
#     )
#     return llm

# def creat_prompt(template):
#     prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", template),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )
#     return prompt

# def read_vectors_db():
#     db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
#     return db

# db = read_vectors_db()

# llm = load_llm(model_dir)

# contextualize_q_prompt = creat_prompt(cq_system_prompt)

# retriever = db.as_retriever()

# history_aware_retriever = create_history_aware_retriever(
#     llm, retriever, contextualize_q_prompt
# )

# qa_prompt = creat_prompt(qa_system_prompt)

# question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# store = {}


# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]

# llm_chain = RunnableWithMessageHistory(
#     rag_chain,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="chat_history",
#     output_messages_key="answer",
# )


# chat_history = []

# def get_response(query):
#     # response = rag_chain.invoke({"input": query, "chat_history": chat_history})

#     response = llm_chain.invoke(
#             {"input": query},
#             config={
#                 "configurable": {"session_id": "test"}
#             },
#             )
#     if isinstance(response, dict) and 'answer' in response:
#         result = response['answer']
#     else:
#         result = response

#     chat_history.extend([HumanMessage(content=query), result])

#     return result

# if __name__ == "__main__":
#     t = True
#     while t:
#         question = input("input:")
#         if question.lower() == "bye":
#             t = False
#         else:
#             response = get_response(question)
#             print(response)
