"""
步驟 2: 文字處理模組

功能：
- 將文字分割成段落
- 建立片段 ID
- 保留階層關係
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

# 引入 PDFParser 相關的 dataclasses，如果需要的話 (在 if __name__ == "__main__": 區塊中)
# from .pdf_parser import Document, PageContent 


@dataclass
class TextChunk:
    """文字片段"""
    chunk_id: str
    doc_id: str
    text: str
    chunk_type: str  # "paragraph" 或 "sentence"
    page_num: int
    parent_id: Optional[str] = None  # 父片段 ID（用於階層關係）
    metadata: dict = field(default_factory=dict)


class TextProcessor:
    """文字處理器"""
    
    @staticmethod
    def split_paragraphs(text: str) -> List[str]:
        """
        將文字分割成段落 (優化：處理 PDF 文本中的換行)
        
        策略：將連續多個空格視為段落分隔，確保句子不被單個換行符打斷。
        """
        # 步驟 1: 將單個換行符號 (在 PDF 提取中常表示行尾) 替換為單個空格
        # 但保留雙換行符號 (可能表示真正的段落分隔)
        text = re.sub(r'(?<!\n)\n(?![\n\s])', ' ', text)
        
        # 步驟 2: 使用雙換行符號或多個連續空格來分割段落
        paragraphs = [p.strip() for p in re.split(r'(\n\n|\s{3,})', text) if p.strip()]
        
        # 最終清理，過濾太短的片段（少於 10 個字符），但保留較長的內容
        # 注意：10 個字符是一個較低的閾值，可以保留更多有效內容
        paragraphs = [p for p in paragraphs if len(p) >= 10] 
        return paragraphs
    
    # 暫時不使用 split_sentences 函數，只專注於段落級別的 Chunk
    
    def process_document(self, document, chunk_size: int = 800, overlap: int = 150) -> List[TextChunk]:
        """
        處理整個文檔，返回所有文字片段（使用重疊窗口以保留更多上下文）
        
        Args:
            document: Document 物件（從 PDFParser 返回）
            chunk_size: 每個 chunk 的最大字符數（默認 500）
            overlap: 重疊的字符數（默認 100，約 20% 重疊）
        
        Returns:
            文字片段列表
        """
        if document is None:
            return []
        
        chunks = []
        
        for page in document.pages:
            # 確保有文本才處理 (在 PDFParser 中已處理，但再次檢查)
            if not page.text or not page.text.strip():
                continue
                
            # 分割成段落
            paragraphs = self.split_paragraphs(page.text)
            
            # 如果段落分割後為空，嘗試使用整個頁面文本作為一個 chunk
            if not paragraphs and len(page.text.strip()) >= 10:
                paragraphs = [page.text.strip()]
            
            # 使用重疊窗口策略
            current_chunk = ""
            para_idx = 0
            chunk_idx = 0
            
            for paragraph in paragraphs:
                # 如果當前 chunk 加上新段落會超過大小，先保存當前 chunk
                if current_chunk and len(current_chunk) + len(paragraph) + 1 > chunk_size:
                    # 保存當前 chunk
                    para_chunk_id = f"{document.doc_id}_p{page.page_num}_chunk{chunk_idx}"
                    para_chunk = TextChunk(
                        chunk_id=para_chunk_id,
                        doc_id=document.doc_id,
                        text=current_chunk.strip(),
                        chunk_type="paragraph",
                        page_num=page.page_num,
                        metadata={"chunk_index": chunk_idx, "paragraph_index": para_idx}
                    )
                    chunks.append(para_chunk)
                    chunk_idx += 1
                    
                    # 保留重疊部分（取最後 overlap 個字符）
                    if len(current_chunk) > overlap:
                        # 在重疊部分找最後一個完整的句子或詞
                        overlap_text = current_chunk[-overlap:]
                        # 嘗試在句子邊界分割
                        sentence_end = max(overlap_text.rfind('. '), overlap_text.rfind('.\n'), overlap_text.rfind(' '))
                        if sentence_end > overlap // 2:  # 如果找到合理的分割點
                            overlap_text = overlap_text[sentence_end+1:]
                        current_chunk = overlap_text + " " + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    # 添加到當前 chunk
                    if current_chunk:
                        current_chunk += " " + paragraph
                    else:
                        current_chunk = paragraph
                
                para_idx += 1
            
            # 保存最後一個 chunk
            if current_chunk.strip():
                para_chunk_id = f"{document.doc_id}_p{page.page_num}_chunk{chunk_idx}"
                para_chunk = TextChunk(
                    chunk_id=para_chunk_id,
                    doc_id=document.doc_id,
                    text=current_chunk.strip(),
                    chunk_type="paragraph",
                    page_num=page.page_num,
                    metadata={"chunk_index": chunk_idx, "paragraph_index": para_idx}
                )
                chunks.append(para_chunk)
                
        return chunks


# 測試程式碼
if __name__ == "__main__":
    from .pdf_parser import Document, PageContent # 使用相對路徑
    
    # 測試文字處理
    processor = TextProcessor()
    
    # 測試段落分割
    test_text = "這是第一段的內容。該句在行尾。\n這是同一段的第二行。  \n\n這是第二段的開始。\n\n這是第三段。"
    paragraphs = processor.split_paragraphs(test_text)
    print(f"段落數: {len(paragraphs)}")
    for i, para in enumerate(paragraphs):
        print(f"段落 {i+1}: {para}")
    
    # 測試文件處理
    doc = Document(
        doc_id="test_doc",
        title="Test Document",
        pages=[PageContent(page_num=1, text=test_text, images=[])],
        metadata={}
    )
    chunks = processor.process_document(doc)
    print(f"\nChunk 數量: {len(chunks)}")
    if chunks:
        print(f"第一個 Chunk: {chunks[0].text}")