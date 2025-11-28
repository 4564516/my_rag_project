"""
使用訓練數據運行並評估
"""
import asyncio
import csv
import sys
import os
from pathlib import Path

# 添加項目路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.evaluate import evaluate
from scripts.embedder import Embedder
from scripts.vector_store import VectorStore
from scripts.llm_client import LLMClient
from scripts.rag_pipeline import RAGPipeline
from scripts.answer_formatter import AnswerFormatter
from scripts.config import Config
from scripts.example_retriever import ExampleRetriever
from scripts.pdf_parser import PDFParser
from scripts.text_processor import TextProcessor

async def run_with_train_data():
    """使用訓練數據運行"""
    # 臨時修改配置使用訓練數據
    config = Config()
    
    # 使用訓練數據作為輸入
    train_csv = project_root / "data/train_QA.csv"
    output_csv = project_root / "artifacts/train_answers.csv"
    
    # 讀取訓練數據的問題
    questions = []
    with train_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("question", "").strip():
                questions.append({
                    "id": row.get("id", ""),
                    "question": row.get("question", "").strip()
                })
    
    print(f"從訓練數據中讀取到 {len(questions)} 個問題")
    
    # 初始化
    print("\n=== 初始化模塊 ===")
    embedder = Embedder(model_name=config.embedding_model)
    
    # 初始化示例檢索器
    print("初始化動態示例檢索器...")
    example_retriever = ExampleRetriever(str(train_csv), embedder)
    
    vector_store = VectorStore(db_path=config.db_path)
    
    # 這裡我們默認不再重建索引，因為 pipeline_step3_index.py 已經建立了最強的索引
    recreate_index = os.getenv("RECREATE_INDEX", "false").lower() == "true"
    vector_store.initialize(config.collection_name, embedder, recreate=recreate_index)
    
    # 初始化 LLM
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
        
        metadata_path = project_root / config.metadata_csv
        formatter = AnswerFormatter(metadata_path=str(metadata_path))
    except ValueError as e:
        print(f"⚠️  警告：無法初始化 LLM: {e}")
        return

    # 檢查索引狀態
    existing_count = 0
    try:
        existing_count = vector_store.get_document_count()
        print(f"向量庫中現有 {existing_count} 個片段")
    except:
        pass

    # Fallback Mode: 只有當索引為空時才跑舊流程
    if existing_count == 0 or recreate_index:
        print("\n=== 解析 PDF 並建立索引 (Fallback Mode) ===")
        print("注意：建議使用 scripts/pipeline_step*.py 獲得更好的效果")
        parser = PDFParser()
        processor = TextProcessor()
        pdf_dir = project_root / config.pdf_dir
        all_chunks = []
        
        if pdf_dir.exists():
            pdf_files = list(pdf_dir.glob("*.pdf"))
            print(f"找到 {len(pdf_files)} 個 PDF 文件")
            
            for pdf_file in pdf_files:
                doc_id = pdf_file.stem
                document = parser.parse(pdf_file, doc_id=doc_id)
                if document:
                    chunks = processor.process_document(document, chunk_size=400, overlap=150)
                    all_chunks.extend(chunks)
            
            if all_chunks:
                print(f"正在為 {len(all_chunks)} 個片段生成 embeddings...")
                texts = [chunk.text for chunk in all_chunks]
                embeddings = embedder.encode(texts)
                vector_store.add_chunks(all_chunks, embeddings)
                print(f"索引建立完成！")
    else:
        print("\n✅ 檢測到現有索引，將直接使用 (跳過 PDF 解析)")
    
    # 回答問題
    print(f"\n=== 回答問題 ===")
    print(f"共 {len(questions)} 個問題需要回答\n")
    
    answers = []
    for idx, q in enumerate(questions, 1):
        question = q["question"]
        print(f"\n[{idx}/{len(questions)}] 問題: {question[:60]}...")
        
        try:
            result = await pipeline.answer(
                question,
                top_k=config.top_k,
                system_prompt=config.system_prompt
            )
            
            await asyncio.sleep(0.5)  # 小延遲
            
        except Exception as e:
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
        
        answers.append({
            "id": q["id"],
            "question": question,
            "answer": formatted["answer"],
            "answer_value": formatted["answer_value"],
            "answer_unit": formatted["answer_unit"],
            "ref_id": formatted["ref_id"],
            "ref_url": formatted["ref_url"],
            "supporting_materials": formatted["supporting_materials"],
            "explanation": formatted["explanation"]
        })
        
        print(f"答案: {formatted['answer'][:60]}...")
        print(f"Value: {formatted['answer_value']}, Unit: {formatted['answer_unit']}")
    
    # 保存答案
    if answers:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["id", "question", "answer", "answer_value", "answer_unit", "ref_id", "ref_url", "supporting_materials", "explanation"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(answers)
        
        print(f"\n已寫入 {len(answers)} 個答案到 {output_csv}")
        
        # 評估答案
        print("\n" + "="*50)
        evaluate(output_csv, train_csv)

if __name__ == "__main__":
    asyncio.run(run_with_train_data())
