"""
步驟 1: PDF 解析模組

功能：
- 從 PDF 檔案提取文字
- 提取圖像資訊（可選）
- 返回結構化的文檔資料
"""

import re # 新增 re 模組用於清理文本
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
from pypdf import PdfReader


@dataclass
class PageContent:
    """單頁的內容"""
    page_num: int
    text: str
    images: List[dict]  # 圖像資訊列表


@dataclass
class Document:
    """完整文檔"""
    doc_id: str
    title: str
    pages: List[PageContent]
    metadata: dict


class PDFParser:
    """PDF 解析器"""
    
    def __init__(self):
        """初始化解析器"""
        pass
    
    def parse(self, pdf_path: Path, doc_id: str, title: str = "") -> Optional[Document]:
        """
        解析 PDF 檔案
        
        Args:
            pdf_path: PDF 檔案路徑
            doc_id: 文檔 ID
            title: 文檔標題（可選）
        
        Returns:
            Document 物件，如果解析失敗則返回 None
        """
        # 檢查檔案是否存在
        if not pdf_path.exists():
            print(f"警告：PDF 檔案不存在於 {pdf_path}")
            return None

        try:
            reader = PdfReader(str(pdf_path))
            total_pages = len(reader.pages)
        except Exception as e:
            print(f"錯誤：無法讀取 PDF 檔案 {pdf_path}: {e}")
            return None
        
        if total_pages == 0:
            print(f"警告：PDF 檔案 {doc_id} 沒有頁面")
            return None
            
        pages = []
        pages_with_text = 0
        
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                # 提取文字
                text = page.extract_text() or ""
                
                # 清理文本：保留換行符（用於段落識別），但規範化多餘的空白
                # 先將多個連續空格/製表符替換為單個空格
                cleaned_text = re.sub(r'[ \t]+', ' ', text)
                # 保留換行符，但將多個連續換行符規範化為最多兩個
                cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
                cleaned_text = cleaned_text.strip()
                
            except Exception as e:
                 print(f"警告：解析 {doc_id} 第 {page_num} 頁時發生錯誤: {e}")
                 cleaned_text = ""

            if not cleaned_text or len(cleaned_text) < 5:
                continue # 忽略空的或太短的頁面

            pages_with_text += 1
            # 圖像提取（目前為空）
            images = []

            pages.append(PageContent(
                page_num=page_num,
                text=cleaned_text,
                images=images
            ))
        
        if not pages:
             print(f"警告：文檔 {doc_id} 未能提取出任何有效文本頁面（總頁數: {total_pages}）。")
             print(f"  這可能是因為 PDF 是掃描版（圖片）或使用了特殊編碼。")
             return None
        
        print(f"  成功提取 {pages_with_text}/{total_pages} 頁的有效文本")

        return Document(
            doc_id=doc_id,
            title=title or doc_id,
            pages=pages,
            metadata={}
        )


# 測試程式碼
if __name__ == "__main__":
    # 測試 PDF 解析
    parser = PDFParser()
    # ⚠️ 請替換成您實際的 PDF 檔案路徑進行測試
    pdf_path = Path("../artifacts/raw_pdfs/example_doc.pdf") 
    
    if pdf_path.exists():
        doc = parser.parse(pdf_path, doc_id="test_doc", title="Test Document")
        if doc:
            print(f"解析完成: {doc.doc_id}")
            print(f"總頁數: {len(doc.pages)}")
            if doc.pages:
                print(f"第一頁文字長度: {len(doc.pages[0].text)}")
                print(f"第一頁文本前 100 字: {doc.pages[0].text[:100]}...")
        else:
             print("文檔解析失敗或無有效內容。")
    else:
        print(f"找不到檔案: {pdf_path}")