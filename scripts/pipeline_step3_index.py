"""
Pipeline Step 3: Indexing Markdown

功能：
- 讀取 artifacts/processed_docs/ 下的 Markdown 文件
- 進行結構化切分 (基於標題)
- 生成 Embeddings (分批處理以避免 OOM)
- 存入 ChromaDB
"""
print("Starting pipeline_step3_index.py...", flush=True)

import re
import sys
from pathlib import Path

# 添加專案根目錄到 sys.path，解決 "No module named 'scripts'" 錯誤
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

print("Importing standard libraries...", flush=True)
from typing import List, Dict
from tqdm import tqdm

print("Importing project modules...", flush=True)
from scripts.embedder import Embedder
from scripts.vector_store import VectorStore
from scripts.config import Config

class MarkdownSplitter:
    """簡單的 Markdown 切分器"""
    
    def split_text(self, text: str) -> List[str]:
        """
        根據標題切分 Markdown
        """
        # 簡單起見，我們按 "## " 切分，保留標題
        chunks = []
        # 使用正則分割，保留分隔符
        parts = re.split(r'(^## .*$)', text, flags=re.MULTILINE)
        
        current_chunk = ""
        for part in parts:
            if part.startswith("## "):
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = part
            else:
                current_chunk += part
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        # 二次處理：如果 chunk 太大 (>1200 字符)，再切
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > 1200:
                # 按段落切分
                paragraphs = chunk.split('\n\n')
                temp_buf = ""
                for p in paragraphs:
                    if len(temp_buf) + len(p) < 1000:
                        temp_buf += "\n\n" + p
                    else:
                        final_chunks.append(temp_buf.strip())
                        temp_buf = p
                if temp_buf:
                    final_chunks.append(temp_buf.strip())
            else:
                final_chunks.append(chunk)
                
        return [c for c in final_chunks if c.strip()]

class SimpleChunk:
    def __init__(self, text, metadata):
        self.text = text
        self.metadata = metadata
        # 從 metadata 提取必要屬性，或給予預設值
        self.chunk_id = str(metadata.get("chunk_id", ""))
        # 確保 chunk_id 是唯一的，加上 doc_id
        if "doc_id" in metadata:
             self.chunk_id = f"{metadata['doc_id']}_{self.chunk_id}"
             
        self.doc_id = metadata.get("doc_id", "unknown")
        self.page_num = metadata.get("page_num", 0)
        self.chunk_type = metadata.get("chunk_type", "markdown_section")
        self.parent_id = metadata.get("parent_id", None)

def index_processed_docs():
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "artifacts/processed_docs"
    
    config = Config()
    print(f"Initializing Embedder: {config.embedding_model}...")
    embedder = Embedder(model_name=config.embedding_model)
    vector_store = VectorStore(db_path=config.db_path)
    
    # 強制重建索引
    print(f"Recreating Vector Store at {config.db_path}...")
    vector_store.initialize(config.collection_name, embedder, recreate=True)
    
    splitter = MarkdownSplitter()
    
    # 獲取所有 Markdown 檔案
    # 修正：使用 rglob 以支援嵌套目錄結構
    md_files = list(processed_dir.rglob("*.md"))
    print(f"Found {len(md_files)} processed Markdown files.")
    
    if not md_files:
        print("❌ No Markdown files found. Please check artifacts/processed_docs/")
        return

    total_chunks = 0
    BATCH_SIZE = 64  # 每 64 個 chunk 存一次，避免記憶體爆炸

    current_batch_texts = []
    current_batch_metadatas = []

    for md_file in tqdm(md_files, desc="Indexing Documents"):
        try:
            # doc_id 通常是父目錄名稱
            doc_id = md_file.parent.name
            content = md_file.read_text(encoding="utf-8")
            
            chunks = splitter.split_text(content)
            # print(f"  {doc_id}: {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks):
                current_batch_texts.append(chunk)
                current_batch_metadatas.append({
                    "doc_id": doc_id,
                    "chunk_id": i,
                    "source": "marker_markdown",
                    "file_path": str(md_file.relative_to(project_root))
                })

                # 當批次滿了，就寫入
                if len(current_batch_texts) >= BATCH_SIZE:
                    embeddings = embedder.encode(current_batch_texts)
                    chunk_objects = [SimpleChunk(t, m) for t, m in zip(current_batch_texts, current_batch_metadatas)]
                    vector_store.add_chunks(chunk_objects, embeddings)
                    
                    total_chunks += len(current_batch_texts)
                    current_batch_texts = []
                    current_batch_metadatas = []
        
        except Exception as e:
            print(f"❌ Error processing {md_file.name}: {e}")
            continue

    # 處理剩餘的 chunks
    if current_batch_texts:
        print(f"Processing final batch of {len(current_batch_texts)} chunks...")
        embeddings = embedder.encode(current_batch_texts)
        chunk_objects = [SimpleChunk(t, m) for t, m in zip(current_batch_texts, current_batch_metadatas)]
        vector_store.add_chunks(chunk_objects, embeddings)
        total_chunks += len(current_batch_texts)

    print(f"\n✅ Indexing Complete! Total chunks indexed: {total_chunks}")
    
    # 驗證
    try:
        count = vector_store.get_document_count()
        print(f"Final Document Count in DB: {count}")
    except:
        pass

if __name__ == "__main__":
    index_processed_docs()
