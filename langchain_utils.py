from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from chroma_utils import vectorstore
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


OPENROUTER_API_KEY =  os.getenv("OPENROUTER_API_KEY")
print(f"Using OpenRouter API Key: {OPENROUTER_API_KEY[:8]}...")
BASE_URL = "https://openrouter.ai/api/v1"

def get_rag_chain(model="deepseek/deepseek-prover-v2"):
    llm = ChatOpenAI(
        model=model,
        api_key=OPENROUTER_API_KEY,
        base_url=BASE_URL,
        max_tokens=200
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# Test function
# def test_rag_chain():
#     rag_chain = get_rag_chain()  # OpenRouter model name
    
#     # Proper chat history (must be LC messages, not tuples)
#     chat_history = [
#         HumanMessage(content="What is LangChain?"),
#         AIMessage(content="LangChain is a framework for building applications with LLMs."),
#     ]
    
#     query = "Can it be used with RAG?"
    
#     response = rag_chain.invoke({
#         "input": query,
#         "chat_history": chat_history
#     })
    
#     print("\n--- Response ---")
#     print(response)

# if __name__ == "__main__":
#     test_rag_chain()

