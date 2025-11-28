"""
步驟 8: 主程式

功能：
- 整合所有步驟
- 實現完整的 WattBot 流程
"""

import asyncio
import csv
import os
from pathlib import Path
from typing import List

from .pdf_parser import PDFParser
from .text_processor import TextProcessor
from .embedder import Embedder
from .vector_store import VectorStore
from .llm_client import LLMClient
from .rag_pipeline import RAGPipeline
from .answer_formatter import AnswerFormatter
from .config import Config
from .example_retriever import ExampleRetriever


async def main():
    """主程式"""
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    config = Config()
    print("=== 載入配置 ===")
    print(f"專案根目錄: {project_root}")
    print(f"PDF 目錄: {config.pdf_dir}")
    print(f"問題檔案: {config.questions_csv}")
    print(f"輸出檔案: {config.output_csv}")
    
    print("\n=== 初始化模組 ===")
    embedder = Embedder(model_name=config.embedding_model)
    
    train_csv_path = project_root / "data/train_QA.csv"
    example_retriever = None
    if train_csv_path.exists():
        print("初始化动态示例检索器...")
        example_retriever = ExampleRetriever(str(train_csv_path), embedder)
    else:
        print("⚠️ 警告：找不到训练数据 data/train_QA.csv，無法使用動態示例。")
        
    vector_store = VectorStore(db_path=config.db_path)
    
    # 這裡我們默認不再重建索引，因為 pipeline_step3_index.py 已經建立了最強的索引
    # 如果用戶想強制重建（例如用舊的 pypdf 方式），需要明確設定 RECREATE_INDEX=true
    recreate_index = os.getenv("RECREATE_INDEX", "false").lower() == "true"
    
    # 初始化 vector store，但不自動加載數據，除非 recreate_index 為 true
    vector_store.initialize(config.collection_name, embedder, recreate=recreate_index)
    
    is_ollama = "ollama" in config.llm_model.lower() or config.llm_model.startswith("ollama/")
    
    if is_ollama:
        if not os.getenv("OPENAI_BASE_URL"):
            os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"
        if not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = "ollama"
        print("✅ 使用 Ollama 本地模型（免費無限制）")
    
    try:
    llm = LLMClient(model=config.llm_model)
        pipeline = RAGPipeline(vector_store, llm, embedder, example_retriever=example_retriever)
        
        # Initialize formatter with metadata path
        metadata_path = project_root / config.metadata_csv
        formatter = AnswerFormatter(metadata_path=str(metadata_path))
        
    except ValueError as e:
        print(f"⚠️  警告：無法初始化 LLM: {e}")
        llm = None
        pipeline = None
        formatter = None
    
    # 檢查索引狀態
    existing_count = 0
    try:
        existing_count = vector_store.get_document_count()
        print(f"向量庫中現有 {existing_count} 個片段")
    except:
        pass

    # 只有在索引為空，或者明確要求重建時，才運行舊的 PDF 解析流程
    # 如果用戶已經跑了 pipeline scripts，這裡 existing_count 應該 > 0，就會跳過這段
    if existing_count == 0 or recreate_index:
        print("\n=== 解析 PDF 並建立索引 (Fallback Mode) ===")
        print("注意：建議使用 scripts/pipeline_step*.py 獲得更好的效果")
        
        parser = PDFParser()
        processor = TextProcessor()
        
        pdf_dir = project_root / config.pdf_dir
        all_chunks = []
        
        if pdf_dir.exists():
            pdf_files = list(pdf_dir.glob("*.pdf"))
            print(f"找到 {len(pdf_files)} 個 PDF 檔案")
            
            if config.limit_pdfs > 0:
                pdf_files = pdf_files[:config.limit_pdfs]
            
            processed_count = 0
            failed_count = 0
            
            for pdf_file in pdf_files:
                doc_id = pdf_file.stem
                print(f"處理: {doc_id}")
                
                document = parser.parse(pdf_file, doc_id=doc_id)
                
                if document is None:
                    print(f"  警告：無法解析 {doc_id}，跳過")
                    failed_count += 1
                    continue
                
                chunks = processor.process_document(document)
                
                if not chunks:
                    print(f"  警告：{doc_id} 未能提取任何文字片段")
                    failed_count += 1
                    continue
                
                print(f"  成功提取 {len(chunks)} 個文字片段")
                all_chunks.extend(chunks)
                processed_count += 1
            
            print(f"\n處理摘要：成功 {processed_count} 個，失敗 {failed_count} 個")
            
            if all_chunks:
                print(f"正在為 {len(all_chunks)} 個片段生成 embeddings...")
                texts = [chunk.text for chunk in all_chunks]
                embeddings = embedder.encode(texts)
                
                vector_store.add_chunks(all_chunks, embeddings)
                print(f"索引建立完成！共 {len(all_chunks)} 個片段")
            else:
                print("警告：雖然找到 PDF 檔案，但未能從中提取任何有效文本片段來建立索引。")
        else:
            print(f"PDF 目錄不存在: {pdf_dir}")
            print("跳過索引建立步驟")
    else:
        print("\n✅ 檢測到現有索引，將直接使用 (跳過 PDF 解析)")
    
    print("\n=== 讀取問題並回答 ===")
    questions_csv = project_root / config.questions_csv
    
    if not questions_csv.exists():
        print(f"問題檔案不存在: {questions_csv}")
        return
    
    if not llm or not pipeline:
        print("LLM 未初始化，退出。")
        return
    
    answers = []
    
    with questions_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        questions = list(reader)
        total_questions = len([q for q in questions if q.get("question", "").strip()])
        
        print(f"共 {total_questions} 個問題需要回答")
        if is_ollama:
            print(f"✅ 使用 Ollama 本地模型，無速率限制\n")
        
        for idx, row in enumerate(questions, 1):
            question = row.get("question", "").strip()
            if not question:
                continue
            
            print(f"\n[{idx}/{total_questions}] 問題: {question[:60]}...")
            
            try:
            result = await pipeline.answer(
                question,
                top_k=config.top_k,
                system_prompt=config.system_prompt
            )
            
                await asyncio.sleep(1)
                
            except (RuntimeError, Exception) as e:
                print(f"❌ 處理問題時發生錯誤: {e}")
                continue
            
            print(f"--- DEBUG: LLM 原始響應 ---")
            print(f"{result['raw_response'][:300]}...")
            print("-" * 40)
            
            formatted = formatter.format_answer(
                result["raw_response"],
                result["ref_ids"],
                question
            )
            
            # 構建符合競賽要求的行
            answer_row = {
                "id": row.get("id", ""),
                "question": question,
                "answer": formatted["answer"],
                "answer_value": formatted["answer_value"],
                "answer_unit": formatted["answer_unit"],
                "ref_id": formatted["ref_id"],
                "ref_url": formatted["ref_url"],
                "supporting_materials": formatted["supporting_materials"],
                "explanation": formatted["explanation"]
            }
            
            answers.append(answer_row)
            
            print(f"答案: {formatted['answer'][:60]}...")
            print(f"Value: {formatted['answer_value']}, Unit: {formatted['answer_unit']}")
    
    if answers:
        output_csv = project_root / config.output_csv
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["id", "question", "answer", "answer_value", "answer_unit", "ref_id", "ref_url", "supporting_materials", "explanation"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(answers)
        
        print(f"\n已寫入 {len(answers)} 個答案到 {output_csv}")


if __name__ == "__main__":
    asyncio.run(main())
