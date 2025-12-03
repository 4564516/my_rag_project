"""
核心配置模組 (Config)
------------------
功能：
1. 集中管理所有系統參數（路徑、模型名稱、參數設置）。
2. 支援從環境變數 (Environment Variables) 覆蓋預設值。
3. 定義 Prompt Template（提示詞模板）。

主要參數：
- EMBEDDING_MODEL: 使用的向量模型 (如 all-mpnet-base-v2)
- LLM_MODEL: 使用的語言模型 (如 ollama/mistral)
- TOP_K: 檢索片段數量
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
    embedding_model: str = "BAAI/bge-m3"  # 升級到最強的開源 Embedding (支持 8192 context, 多語言, 稀疏檢索)
    # LLM 模型選項：
    # - 本地模型（推薦，免費無限制）: "ollama/qwen2.5:14b" (5070 12GB VRAM 推薦)
    # - OpenRouter 免費: "mistralai/mistral-7b-instruct:free" (每天 50 次)
    # - Groq 免費: "llama-3.1-8b-instant" (每分鐘 30 次)
    llm_model: str = "ollama/qwen2.5:14b"  # 升級到 14B，邏輯推理更強
    
    # RAG 配置
    top_k: int = 150  # 大幅增加檢索數量 (從 50 -> 150)，讓 Reranker 有更多選擇
    llm_top_k: int = 15  # 增加最終上下文數量 (從 10 -> 15)
    limit_pdfs: int = 0  # 0 = 處理所有 PDF，>0 = 只處理前 N 個
    rerank_model: str = "BAAI/bge-reranker-v2-m3"  # 換成 SOTA 的 Reranker 模型，大幅提升排序準確度
    
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

