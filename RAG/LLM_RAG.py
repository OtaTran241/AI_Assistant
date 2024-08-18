from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.prompts import MessagesPlaceholder
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

model_dir = "/content/Models/llama-7b-chat_q5_0.gguf"

embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

vector_db_path = "data"

n_gpu_layers = -1

n_batch = 512

qa_system_prompt = """You are an assistant for question-answering tasks. Your name is Toka. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say you don't know, don't try to make one up. Don't make a new question\n
{context}"""


cq_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history. \
Don't make a new question.\
If you Don't know the answer, say you Don't know, Don't try to make one up."""

def load_llm(model_file):
    llm = LlamaCpp(
        model_path=model_file,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        callback_manager=callback_manager,
        verbose=True,
        max_tokens=1024,
        temperature=0.01,
        n_ctx=1024,
    )
    return llm

def creat_prompt(template):
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
    return prompt

def read_vectors_db():
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

db = read_vectors_db()

llm = load_llm(model_dir)

contextualize_q_prompt = creat_prompt(cq_system_prompt)

retriever = db.as_retriever()

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_prompt = creat_prompt(qa_system_prompt)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

llm_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


chat_history = []

def get_response(query):
    # response = rag_chain.invoke({"input": query, "chat_history": chat_history})

    response = llm_chain.invoke(
            {"input": query},
            config={
                "configurable": {"session_id": "test"}
            },
            )
    if isinstance(response, dict) and 'answer' in response:
        result = response['answer']
    else:
        result = response

    chat_history.extend([HumanMessage(content=query), result])

    return result

if __name__ == "__main__":
    t = True
    while t:
        question = input("input:")
        if question.lower() == "bye":
            t = False
        else:
            response = get_response(question)
            print(response)
