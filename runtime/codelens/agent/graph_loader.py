import pickle
import os
from langchain_core.documents import Document
import config

graphs = {}          # repo_name -> G
vectorstores = {}    # repo_name -> FAISS

def get_repo_name(org_repo: str):
    return org_repo.replace("/", "__")

def load_graph():
    with open(config.GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    return G

def build_rag_documents(G):
    docs = []
    for node_id, data in G.nodes(data=True):
        text = f"文件: {data.get('file','')} 行: {data.get('line','')} 类型: {data.get('type','')} 代码片段: {data.get('code','')[:800]}"
        metadata = {**data, "node_id": node_id}
        docs.append(Document(page_content=text, metadata=metadata))
    return docs

def load_graph_for_repo(repo_key: str):
    if repo_key in graphs:
        return graphs[repo_key]
    pkl_path = f"/home/zzz/projects/1.2/Jlocagent/resultgraphs/multi-swe-bench/jackson-dataformat-xml-3.x/graph.pkl"
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"缺少图：{pkl_path} 请先生成")
    with open(pkl_path, "rb") as f:
        graphs[repo_key] = pickle.load(f)
    return graphs[repo_key]