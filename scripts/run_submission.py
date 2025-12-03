"""
Pipeline Step 5: 生成競賽提交檔案 (Run Submission)
-----------------------------------------------
功能：
1. 讀取測試問題集 (`data/test_Q.csv`)。
2. 執行完整的 RAG 流程。
3. 嚴格按照 WattBot 評分標準格式化輸出。
4. 生成 `artifacts/submission.csv`。

WattBot 規則：
- is_NA (10%): 如果無法回答，所有欄位必須為 is_blank。
- answer_value (75%): 數值誤差 < 0.1%。
- ref_id (15%): Jaccard overlap。
"""

import asyncio
import csv
import sys
import os
from pathlib import Path

# 添加項目路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.core.embedder import Embedder
from scripts.core.vector_store import VectorStore
from scripts.core.llm_client import LLMClient
from scripts.core.rag_pipeline import RAGPipeline
from scripts.core.answer_formatter import AnswerFormatter
from scripts.core.config import Config
from scripts.core.example_retriever import ExampleRetriever

def is_true_false_question(question: str) -> bool:
    q = question.strip().lower()
    return q.startswith("true or false") or q.startswith("true/false")

# 專用的 True/False 系統提示
BOOLEAN_SYSTEM_PROMPT = """You are an expert research assistant specialized in answering True/False questions from academic documents.

RULES:
- You will receive a QUESTION and a CONTEXT.
- The CONTEXT may contain quotes, tables, or figure descriptions.
- Your job is to decide if the statement in the QUESTION is TRUE or FALSE based ONLY on the CONTEXT.

OUTPUT RULES:
- If the statement is clearly supported by the CONTEXT → answer_value = "1"
- If the statement is clearly contradicted by the CONTEXT → answer_value = "0"
- Only if the CONTEXT truly has NO information about the statement → answer_value = "is_blank"
- Do NOT answer "is_blank" if there is any explicit sentence that implies True or False.
- Do NOT use outside knowledge. Use ONLY the given CONTEXT.

FORMAT:
- For True/False questions, you MUST:
  - Set "answer_value" to "1" or "0" or "is_blank"
  - Optionally explain in natural language in "answer" and "explanation".
"""

async def run_submission():
    config = Config()
    
    # 輸入與輸出路徑
    test_csv = project_root / "data/test_Q.csv"
    output_csv = project_root / "artifacts/submissionZZZZZ.csv"
    train_csv_for_examples = project_root / "data/train_QA.csv" # 用於 Few-Shot 檢索
    
    if not test_csv.exists():
        print(f"❌ 錯誤：找不到測試檔案 {test_csv}")
        return

    # 讀取測試問題
    questions = []
    with test_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("question", "").strip():
                questions.append({
                    "id": row.get("id", ""),
                    "question": row.get("question", "").strip()
                })
    
    print(f"讀取到 {len(questions)} 個測試問題")
    
    # 初始化模塊
    print("\n=== 初始化模塊 ===")
    embedder = Embedder(model_name=config.embedding_model)
    
    # 使用訓練數據作為 Few-Shot 來源
    print("初始化動態示例檢索器 (使用 train_QA.csv)...")
    if train_csv_for_examples.exists():
        example_retriever = ExampleRetriever(str(train_csv_for_examples), embedder)
    else:
        print("⚠️ 警告：找不到 train_QA.csv，將無法使用動態示例。")
        example_retriever = None
    
    vector_store = VectorStore(db_path=config.db_path)
    vector_store.initialize(config.collection_name, embedder, recreate=False)
    
    # 檢查索引
    try:
        count = vector_store.get_document_count()
        print(f"向量庫中現有 {count} 個片段")
        if count == 0:
            print("❌ 錯誤：向量庫為空！請先執行 pipeline_step3_index.py")
            return
    except:
        pass

    llm = LLMClient(model=config.llm_model)
    print(f"✅ LLM 模型已加載: {config.llm_model}") # 明確顯示當前使用的模型
    pipeline = RAGPipeline(
        vector_store, 
        llm, 
        embedder, 
        example_retriever=example_retriever,
        rerank_model=config.rerank_model
    )
    
    metadata_path = project_root / config.metadata_csv
    formatter = AnswerFormatter(metadata_path=str(metadata_path))
    
    # 開始回答
    print(f"\n=== 開始生成提交檔案 (併發模式) ===")
    
    # 使用 Semaphore 限制併發數 (建議: 14B模型用3-5個，7B模型用5-10個)
    sem = asyncio.Semaphore(5) 
    
    async def process_question(idx, q):
        async with sem:
            question = q["question"]
            print(f"[{idx}/{len(questions)}] 正在處理 ID: {q['id']}...")
            
            try:
                # 選擇 Prompt
                if is_true_false_question(question):
                    system_prompt = BOOLEAN_SYSTEM_PROMPT
                else:
                    system_prompt = config.system_prompt

                result = await pipeline.answer(
                    question,
                    top_k=config.top_k,  # 使用 Config 統一管理
                    llm_top_k=config.llm_top_k, # 使用 Config 統一管理
                    system_prompt=system_prompt
                )
                
                formatted = formatter.format_answer(
                    result["raw_response"],
                    result["ref_ids"],
                    question
                )
                
                # WattBot 規則
                if formatted["answer_value"] == "is_blank":
                    formatted["answer_unit"] = "is_blank"
                    formatted["ref_id"] = "is_blank"
                    formatted["ref_url"] = "is_blank"
                    formatted["supporting_materials"] = "is_blank"
                
                print(f"  ✅ [{idx}] 完成: {formatted['answer_value'][:20]}")
                
                return {
                    "id": q["id"],
                    "question": question,
                    "answer": formatted["answer"],
                    "answer_value": formatted["answer_value"],
                    "answer_unit": formatted["answer_unit"],
                    "ref_id": formatted["ref_id"],
                    "ref_url": formatted["ref_url"],
                    "supporting_materials": formatted["supporting_materials"],
                    "explanation": formatted["explanation"]
                }
                
            except Exception as e:
                print(f"  ❌ [{idx}] 錯誤: {e}")
                return {
                    "id": q["id"],
                    "question": question,
                    "answer": "Error processing question",
                    "answer_value": "is_blank",
                    "answer_unit": "is_blank",
                    "ref_id": "is_blank",
                    "ref_url": "is_blank",
                    "supporting_materials": "is_blank",
                    "explanation": str(e)
                }

    # 建立所有任務
    tasks = [process_question(i, q) for i, q in enumerate(questions, 1)]
    
    # 併發執行
    results = await asyncio.gather(*tasks)
    answers = results

    # 寫入 CSV
    if answers:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            # 確保欄位順序符合提交要求
            fieldnames = ["id", "question", "answer", "answer_value", "answer_unit", "ref_id", "ref_url", "supporting_materials", "explanation"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(answers)
        
        print(f"\n✅ 原始提交檔案已生成: {output_csv}")
        
        # --- 自動修復邏輯 (Auto-Fix) ---
        print("\n=== 正在執行最終檢查與修復 (Auto-Fix) ===")
        try:
            import pandas as pd
            
            # 1. 讀取並強制轉為字串
            df = pd.read_csv(output_csv, dtype=str, keep_default_na=False)
            
            # 2. 核彈級空值填補
            df = df.replace(r'^\s*$', 'is_blank', regex=True)
            for col in df.columns:
                if df[col].dtype == object:
                    mask = df[col].astype(str).str.lower().isin(['nan', 'none'])
                    df.loc[mask, col] = 'is_blank'
            df = df.fillna('is_blank')
            
            # 3. 強制 ID 順序 (從 test_Q.csv 讀取正確 ID)
            print("正在校對 ID 順序...")
            test_ids = []
            with test_csv.open("r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("id"):
                        test_ids.append(row["id"].strip())
            
            # 確保行數一致
            if len(df) == len(test_ids):
                df['id'] = test_ids
                print("ID 校對完成。")
            else:
                print(f"⚠️ 警告：生成行數 ({len(df)}) 與測試集行數 ({len(test_ids)}) 不符，跳過 ID 校對。")
            
            # 4. 再次檢查 is_NA 規則
            mask = df['answer_value'].str.lower() == 'is_blank'
            cols_to_blank = ['answer_unit', 'ref_id', 'ref_url', 'supporting_materials']
            for col in cols_to_blank:
                df.loc[mask, col] = 'is_blank'
                
            # 5. 覆蓋寫入
            df.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"✅ 最終修正版已覆蓋至: {output_csv}")
            print("此檔案已準備好上傳 Kaggle。")
            
        except ImportError:
            print("⚠️ 警告：找不到 pandas，無法執行高級修復。請手動檢查 CSV。")
        except Exception as e:
            print(f"⚠️ 修復過程發生錯誤: {e}")

if __name__ == "__main__":
    asyncio.run(run_submission())

