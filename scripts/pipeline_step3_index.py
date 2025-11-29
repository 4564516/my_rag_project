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
from scripts.core.embedder import Embedder
from scripts.core.vector_store import VectorStore
from scripts.core.config import Config

class MarkdownSplitter:
    """簡單的 Markdown 切分器"""
    
    def split_text(self, text: str) -> List[str]:
        """
        根據標題切分 Markdown，確保表格和圖表作為完整單元
        """
        # 第一步：按 "## " 標題切分
        chunks = []
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
            
        # 第二步：處理大 chunk，但保護表格和圖表
        final_chunks = []
        for chunk in chunks:
            # 特殊切分：強制將 **Figure Data (Q&A):** 和 **Figure Data (Table):** 分開
            
            # 我們使用 regex 識別這些新的標記
            # 1. **Figure Data (Q&A):**
            # 2. **Figure Data (Table):**
            
            # 先按 Q&A 分割
            sub_parts = re.split(r'(?=\*\*Figure Data \(Q&A\):\*\*)', chunk)
            
            temp_parts = []
            for sub_part in sub_parts:
                # 再按 Table 分割
                sub_sub_parts = re.split(r'(?=\*\*Figure Data \(Table\):\*\*)', sub_part)
                temp_parts.extend(sub_sub_parts)
            
            current_figure_context = ""
            
            for sub_part in temp_parts:
                if not sub_part.strip():
                    continue
                
                # 嘗試提取 Context
                if "**Figure Context:**" in sub_part:
                    current_figure_context = sub_part.replace("**Figure Context:**", "").strip()[:500]
                
                # 處理 Q&A 部分：注入 Context (如果需要，但 Q&A 本身已經很完整了，其實不注入也可以，但注入更安全)
                text_to_process = sub_part
                
                # 如果是 Q&A 區塊，我們特別標記它，讓它保持完整
                if "**Figure Data (Q&A):**" in sub_part:
                    # 注入 Context 以防萬一
                    if current_figure_context:
                        text_to_process = f"[Context: {current_figure_context}]\n" + sub_part
                    
                    final_chunks.append(text_to_process)
                    continue # 跳過後續切分，直接作為一個完整 chunk
                
                # 一般內容的處理（包括 Table 和正文）
                if len(text_to_process) > 1200:
                    # 按段落切分，但保護表格和圖表
                    lines = text_to_process.split('\n')
                    temp_buf = ""
                    in_table = False
                    in_figure = False
                    
                    for i, line in enumerate(lines):
                        # 檢測表格開始（Markdown 表格行）
                        if line.strip().startswith('|') and '|' in line[1:]:
                            in_table = True
                        # 檢測表格結束（空行或非表格行）
                        elif in_table and not line.strip().startswith('|') and line.strip():
                            in_table = False
                        
                        # 檢測 Figure 相關標記
                        if "**Figure" in line or "Table Processing" in line:
                            in_figure = True
                        elif in_figure and (line.strip().startswith('![') or line.strip().startswith('#')):
                            in_figure = False
                        
                        # 如果當前在表格或圖表中，強制加入當前 buffer
                        if in_table or in_figure:
                            temp_buf += "\n" + line
                            # 只有當 buffer 極度大（>4000）時才強制切分
                            if len(temp_buf) > 4000 and not in_table:
                                final_chunks.append(temp_buf.strip())
                                temp_buf = line
                        else:
                            # 正常處理
                            if len(temp_buf) + len(line) < 600: 
                                temp_buf += "\n" + line
                            else:
                                if temp_buf.strip():
                                    final_chunks.append(temp_buf.strip())
                                temp_buf = line
                    
                    if temp_buf.strip():
                        final_chunks.append(temp_buf.strip())
                else:
                    final_chunks.append(text_to_process)
                
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
