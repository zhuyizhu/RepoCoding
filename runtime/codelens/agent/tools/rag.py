from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from agent.graph_loader import load_graph, build_rag_documents
import config
import os

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")  # 超强中英+代码

vectorstore = None

def init_rag():
    global vectorstore
    if vectorstore is None:
        if os.path.exists(config.VECTORSTORE_PATH):
            vectorstore = FAISS.load_local(config.VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
        else:
            G = load_graph()
            docs = build_rag_documents(G)
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(config.VECTORSTORE_PATH)
    return vectorstore

@tool
def hybrid_rag_search(query: str, k: int = 15) -> str:
    """语义搜索整个 Java 项目（和图搜索互补）"""
    vs = init_rag()
    docs = vs.similarity_search(query, k=k)
    return "\n\n".join([f"【匹配】{doc.page_content[:600]}" for doc in docs])