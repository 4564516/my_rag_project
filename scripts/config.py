"""
配置檔案

集中管理所有配置參數
"""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """配置類別"""
    
    # 路徑配置
    pdf_dir: str = "artifacts/raw_pdfs"
    questions_csv: str = os.getenv("QUESTIONS_CSV", "data/test_Q.csv")
    output_csv: str = os.getenv("OUTPUT_CSV", "artifacts/answers.csv")
    metadata_csv: str = "data/metadata.csv"  # 新增 metadata 路徑
    db_path: str = "./chroma_db"
    collection_name: str = "wattbot_index"
    
    # 模型配置
    embedding_model: str = "all-mpnet-base-v2"  # 升级到更好的 embedding 模型（768 维，更准确）
    # LLM 模型選項：
    # - 本地模型（推薦，免費無限制）: "ollama/llama3.2:3b" 或 "ollama/mistral:7b"
    # - OpenRouter 免費: "mistralai/mistral-7b-instruct:free" (每天 50 次)
    # - Groq 免費: "llama-3.1-8b-instant" (每分鐘 30 次)
    llm_model: str = "ollama/mistral:7b"  # 默認使用 Ollama 本地模型（更大的模型，更準確）
    
    # RAG 配置
    top_k: int = 50  # 檢索的片段數量（增加到 50 以獲取更多上下文）
    limit_pdfs: int = 0  # 0 = 處理所有 PDF，>0 = 只處理前 N 個
    
    # Prompt 配置
    system_prompt: str = """You are an expert research assistant specialized in extracting precise information from academic documents. 
- Extract EXACT values as they appear in the context (numbers, names, true/false)
- For True/False questions, convert to "1" (True) or "0" (False)
- For answer_value: provide ONLY the value (no units, no descriptions)
- If information is not in the context, use "is_blank"
- Use ONLY information explicitly stated in the context"""
    
    def __post_init__(self):
        """後處理：從環境變數覆蓋配置"""
        # 可以從環境變數讀取配置
        self.llm_model = os.getenv("LLM_MODEL", self.llm_model)
        self.top_k = int(os.getenv("TOP_K", self.top_k))

