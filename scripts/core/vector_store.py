"""
向量資料庫與檢索模組 (Vector Store)
---------------------------------
功能：
1. 封裝 ChromaDB 操作 (新增、查詢、刪除)。
2. 實現混合檢索邏輯 (Hybrid Search)：
   - 向量相似度 (Cosine Similarity)
   - 關鍵字加權 (Keyword Boosting)：對表格引用、圖表引用、專有名詞進行加分。
3. 管理 Metadata 過濾。
"""

from typing import List, Optional
import re
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
        
        # 抓取 5 倍候選，或至少 300 個 (MPNet 需要更大的池子來過濾)
        candidate_pool_size = max(top_k * 5, 300)
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=candidate_pool_size
        )
        
        search_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            ids = results['ids'][0]
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            dists = results['distances'][0] if 'distances' in results else [0]*len(ids)
            
            # 提取查詢中的關鍵特徵 (數字、大寫專有名詞、表格/圖表引用)
            query_tokens = query.split()
            special_tokens = []
            table_refs = []  # 表格引用，如 "Table 3"
            figure_refs = []  # 圖表引用，如 "Figure 13"
            
            for t in query_tokens:
                # 清理標點符號
                clean_t = "".join(c for c in t if c.isalnum() or c in "-_")
                
                # 檢測表格/圖表引用
                if "table" in clean_t.lower():
                    # 提取表格編號
                    table_num_match = re.search(r'table\s*(\d+)', query, re.IGNORECASE)
                    if table_num_match:
                        table_refs.append(f"table {table_num_match.group(1)}")
                
                if "figure" in clean_t.lower() or "fig" in clean_t.lower():
                    # 提取圖表編號
                    fig_num_match = re.search(r'figure\s*(\d+)|fig\s*\.?\s*(\d+)', query, re.IGNORECASE)
                    if fig_num_match:
                        fig_num = fig_num_match.group(1) or fig_num_match.group(2)
                        figure_refs.append(f"figure {fig_num}")
                
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
            # print(f"DEBUG: Table Refs: {table_refs}, Figure Refs: {figure_refs}")

            for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
                base_score = 1.0 - dist
                
                # 關鍵字加權
                boost = 0.0
                text_lower = text.lower()
                
                # 1. 表格/圖表引用匹配（高優先級）
                for table_ref in table_refs:
                    if table_ref in text_lower:
                        boost += 0.3  # 表格引用匹配給予高權重
                        # 如果還包含表格標記，額外加分
                        if "table" in text_lower and any(c.isdigit() for c in text):
                            boost += 0.2
                
                for figure_ref in figure_refs:
                    if figure_ref in text_lower:
                        boost += 0.3  # 圖表引用匹配給予高權重
                        # 如果還包含圖表描述，額外加分
                        if "figure description" in text_lower:
                            boost += 0.2
                
                # 2. 表格標記匹配（即使沒有明確引用）
                if "table" in query.lower() and ("table" in text_lower and "|" in text):
                    boost += 0.25  # 檢測到表格結構
                
                # 3. 關鍵字匹配
                matched_count = 0
                for token in special_tokens:
                    if token in text_lower:
                        boost += 0.15  # 命中一個關鍵特徵加 0.15 分
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
