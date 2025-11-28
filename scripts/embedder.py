"""
步驟 3: Embedding 模組

功能：
- 載入 embedding 模型
- 將文字轉換成向量
- 批次處理
"""

import os
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

# 修復 tokenizers 並行性警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Embedder:
    """Embedding 生成器"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化 Embedder
        
        Args:
            model_name: sentence-transformers 模型名稱
                       推薦選項：
                       - "all-MiniLM-L6-v2" (快，384維)
                       - "all-mpnet-base-v2" (準，768維)
        """
        print(f"載入 embedding 模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"模型維度: {self.dimension}")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        將文字列表轉換成 embeddings
        
        Args:
            texts: 文字列表
        
        Returns:
            numpy array，shape: (len(texts), dimension)
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        將單個文字轉換成 embedding
        
        Args:
            text: 單個文字
        
        Returns:
            numpy array，shape: (dimension,)
        """
        return self.encode([text])[0]


# 測試程式碼
if __name__ == "__main__":
    # 測試 Embedding
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    
    test_texts = [
        "這是第一段文字",
        "這是第二段文字",
        "What is RAG?"
    ]
    
    embeddings = embedder.encode(test_texts)
    print(f"輸入文字數: {len(test_texts)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"第一個 embedding 的前 5 個值: {embeddings[0][:5]}")

