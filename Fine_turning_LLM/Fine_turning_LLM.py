from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

model_dir = "models/llama-7b-chat_q5_0.gguf"

embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

vector_db_path = "data/db_faiss"

n_gpu_layers = -1

n_batch = 512

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
    prompt = PromptTemplate(template = template, input_variables=["context", "question"])
    return prompt

def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = db.as_retriever(search_kwargs={"k":4}),
        return_source_documents = False,
        chain_type_kwargs= {'prompt': prompt}
    )

    return llm_chain

def read_vectors_db():
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization = True)
    return db

db = read_vectors_db()

llm = load_llm(model_dir)

# template = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  
#             Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
#             Please ensure that your responses are socially unbiased and positive in nature. 
#             If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
#             If you don't know the answer to a question, please don't share false information.

#             context: {context}
#             question: {question}

#             Answer the question and provide additional helpful information,
#             based on the pieces of information, if applicable. Be succinct.

#             Responses should be properly formatted to be easily read.
#             """

template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""

prompt = creat_prompt(template)

llm_chain  = create_qa_chain(prompt, llm, db)

def get_response(query):
    response = llm_chain.invoke({"query": query})
    
    return response['result']

if __name__ == "__main__":
    t = True
    while t:
        question = input("input:")
        if question == "bye":
            t = False
        else:
            response = llm_chain.invoke({"query": question})
            print(response["result"])