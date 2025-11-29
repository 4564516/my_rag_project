"""
Pipeline Step 4: 訓練與評估 (Run Training & Evaluation)
---------------------------------------------------
功能：
1. 讀取訓練問題集 (`data/train_QA.csv`)。
2. 執行完整的 RAG 流程 (Query Rewrite -> Retrieval -> Rerank -> Generation)。
3. 將生成的答案寫入 `artifacts/train_answers.csv`。
4. 自動調用 `evaluate.py` 計算準確率。

執行方式：
python scripts/run_train.py
"""

import asyncio
import csv
from pathlib import Path
import sys
import os

# 添加項目路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.core.evaluate import evaluate
from scripts.core.embedder import Embedder
from scripts.core.vector_store import VectorStore
from scripts.core.llm_client import LLMClient
from scripts.core.rag_pipeline import RAGPipeline
from scripts.core.answer_formatter import AnswerFormatter
from scripts.core.config import Config
from scripts.core.example_retriever import ExampleRetriever

def is_true_false_question(question: str) -> bool:
    """
    判斷是否為 True/False 題目。
    目前根據開頭是否包含 'True or False' 來判斷。
    """
    q = question.strip().lower()
    return q.startswith("true or false") or q.startswith("true/false")


# 專用的 True/False 系統提示，強制輸出 1 / 0 / is_blank
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

    if existing_count == 0:
        print("❌ 錯誤：向量庫為空！")
        print("請先執行以下指令建立索引：")
        print("  1. python scripts/pipeline_step1_marker.py")
        print("  2. python scripts/pipeline_step2_vision.py")
        print("  3. python scripts/pipeline_step3_index.py")
        return
    else:
        print("\n✅ 檢測到現有索引，開始執行 RAG...")
    
    # 回答問題
    print(f"\n=== 回答問題 ===")
    print(f"共 {len(questions)} 個問題需要回答\n")
    
    answers = []
    for idx, q in enumerate(questions, 1):
        question = q["question"]
        print(f"\n[{idx}/{len(questions)}] 問題: {question[:60]}...")
        
        try:
            # 根據題目類型選擇系統提示：
            # True/False 題使用更嚴格的布林判斷提示，避免隨便回答 is_blank。
            if is_true_false_question(question):
                system_prompt = BOOLEAN_SYSTEM_PROMPT
            else:
                system_prompt = config.system_prompt

            result = await pipeline.answer(
                question,
                top_k=config.top_k,
                llm_top_k=config.llm_top_k,
                system_prompt=system_prompt
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
