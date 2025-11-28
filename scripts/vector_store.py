"""
步驟 4: 向量儲存模組

功能：
- 儲存文檔片段和對應的向量
- 相似度搜尋 (Vector Search)
- 關鍵字加權 (Simple Keyword Boosting)
- 返回 top-k 結果
"""

from typing import List, Optional
import chromadb
from chromadb.config import Settings
import numpy as np

class VectorStore:
    """向量儲存與檢索 (Vector + Simple Keyword Boosting)"""
    
    def __init__(self, db_path: str = "./chroma_db"):
        """
        初始化向量庫
        
        Args:
            db_path: ChromaDB 資料庫路徑
        """
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = None
        self.embedder = None
    
    def initialize(self, collection_name: str, embedder, recreate: bool = False):
        """
        初始化 collection 和 embedder
        """
        self.embedder = embedder
        
        # 檢查 collection 是否已存在
        try:
            existing_collections = [c.name for c in self.client.list_collections()]
            if collection_name in existing_collections:
                if recreate:
                    # 如果要求重建，刪除舊的
                    self.client.delete_collection(collection_name)
                    print(f"已刪除舊的 collection: {collection_name}")
                else:
                    # 復用現有的 collection
                    self.collection = self.client.get_collection(name=collection_name)
                    count = self.collection.count()
                    print(f"復用現有的 collection: {collection_name} (已有 {count} 個片段)")
                    return
        except Exception as e:
            print(f"初始化錯誤: {e}")
            pass
        
        # 建立新的 collection
        if not self.collection:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # 使用 cosine 相似度
            )
            print(f"已建立 collection: {collection_name}")
    
    def add_chunks(self, chunks: List, embeddings: np.ndarray):
        """
        將文檔片段和 embeddings 加入向量庫
        """
        if not self.embedder:
            raise ValueError("請先呼叫 initialize()")
        
        if len(chunks) != len(embeddings):
            raise ValueError("chunks 和 embeddings 數量不匹配")
        
        # 準備資料
        texts = [chunk.text for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        
        # 準備 metadata
        metadatas = []
        for chunk in chunks:
            meta = {
                "doc_id": chunk.doc_id,
                "page_num": chunk.page_num,
                "chunk_type": chunk.chunk_type,
            }
            if chunk.parent_id:
                meta["parent_id"] = chunk.parent_id
            if chunk.metadata:
                meta.update(chunk.metadata)
            metadatas.append(meta)
        
        # 轉換成列表（ChromaDB 需要）
        embeddings_list = embeddings.tolist()
        
        # 加入 ChromaDB
        self.collection.add(
            embeddings=embeddings_list,
            documents=texts,
            ids=ids,
            metadatas=metadatas
        )
        print(f"已加入 {len(chunks)} 個片段到向量庫")
    
    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """
        搜尋最相關的片段 (Simple Hybrid: Vector + String Matching Boosting)
        """
        if not self.embedder:
            raise ValueError("請先呼叫 initialize()")
        
        # 1. Vector Search (獲取候選)
        query_embedding = self.embedder.encode_single(query)
        # 抓取 3 倍候選，確保有足夠的空間讓關鍵字匹配的項目浮上來
        vector_k = min(top_k * 3, 100)
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=vector_k
        )
        
        search_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            ids = results['ids'][0]
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            dists = results['distances'][0] if 'distances' in results else [0]*len(ids)
            
            # 提取查詢中的關鍵特徵 (數字、大寫專有名詞)
            query_tokens = query.split()
            special_tokens = []
            for t in query_tokens:
                # 清理標點符號
                clean_t = "".join(c for c in t if c.isalnum() or c in "-_")
                
                # 條件：
                # 1. 包含數字 (如 5,842, 2023)
                # 2. 大寫開頭且長度 > 3 (如 GShard, LLaMA, Python)
                # 3. 全大寫 (如 AI, GPT)
                if (any(c.isdigit() for c in clean_t) and len(clean_t)>1) or \
                   (clean_t and clean_t[0].isupper() and len(clean_t)>3) or \
                   (clean_t.isupper() and len(clean_t)>1):
                    special_tokens.append(clean_t.lower())
            
            # 移除常見干擾詞
            stop_words = {"what", "when", "where", "which", "who", "how", "does", "that", "this", "with", "from", "about", "true", "false"}
            special_tokens = [t for t in special_tokens if t not in stop_words]
            
            # print(f"DEBUG: Query Special Tokens: {special_tokens}")

            for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
                base_score = 1.0 - dist
                
                # 關鍵字加權
                boost = 0.0
                text_lower = text.lower()
                
                matched_count = 0
                for token in special_tokens:
                    if token in text_lower:
                        boost += 0.15  # 命中一個關鍵特徵加 0.15 分 (顯著權重)
                        matched_count += 1
                
                # 額外獎勵：如果命中了多個關鍵特徵，給予額外加分
                if matched_count >= 2:
                    boost += 0.1
                
                final_score = base_score + boost
                
                search_results.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "score": final_score,
                    "metadata": meta,
                    "vector_score": base_score,
                    "boost": boost
                })
        
        # 排序並返回 Top K
        search_results.sort(key=lambda x: x['score'], reverse=True)
        return search_results[:top_k]

    def get_document_count(self) -> int:
        """獲取文檔數量"""
        if not self.collection:
            return 0
        return self.collection.count()

# 測試程式碼
if __name__ == "__main__":
    pass
