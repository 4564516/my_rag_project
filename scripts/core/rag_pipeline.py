"""
RAG 流程模組 (RAG Pipeline)
-------------------------
功能：
1. 整合 VectorStore (檢索)、Reranker (重排序)、LLMClient (生成)。
2. 實現 RAG 的標準流程：
   - Query Rewrite (查詢重寫/擴展)
   - Retrieval (初步檢索)
   - Reranking (使用 Cross-Encoder 精確排序)
   - Context Assembly (上下文組裝)
   - Generation (LLM 回答)
3. 支援 Few-Shot Examples (動態示例注入)。
"""

from typing import List, Optional, TYPE_CHECKING
try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False
    print("Warning: sentence-transformers not installed, reranking disabled.")

if TYPE_CHECKING:
    from .vector_store import VectorStore
    from .llm_client import LLMClient
    from .embedder import Embedder
    from .example_retriever import ExampleRetriever


class RAGPipeline:
    """RAG Pipeline - 整合檢索和生成"""
    
    def __init__(
        self,
        vector_store,
        llm,
        embedder,
        example_retriever=None,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        初始化 RAG Pipeline
        
        Args:
            vector_store: 向量儲存實例
            llm: LLM 客戶端實例
            embedder: Embedder 實例
            example_retriever: ExampleRetriever 實例 (可選)
            rerank_model: Rerank 模型名稱
        """
        self.vector_store = vector_store
        self.llm = llm
        self.embedder = embedder
        self.example_retriever = example_retriever
        
        self.reranker = None
        if HAS_CROSS_ENCODER:
            try:
                print(f"Loading Reranker model: {rerank_model}...")
                # 強制使用 CUDA，並傳遞 trust_remote_code
                self.reranker = CrossEncoder(
                    rerank_model, 
                    device='cuda', 
                    automodel_args={"trust_remote_code": True}
                )
                print("Reranker loaded successfully.")
            except Exception as e:
                print(f"Failed to load Reranker: {e}")
    
    async def rewrite_query(self, question: str) -> str:
        """
        使用 LLM 重寫查詢，生成一個假設性的答案段落，以增強檢索效果。
        """
        if not self.example_retriever:
            return question

        # 檢索相關示例來指導重寫
        examples = self.example_retriever.retrieve(question, k=3)
        examples_text = ""
        if examples:
            for i, ex in enumerate(examples, 1):
                examples_text += f"Question: {ex['question']}\nHypothetical Answer Passage: {ex['explanation']}\n\n"

        prompt = f"""You are an expert research assistant. Your task is to write a hypothetical answer passage for the given question. 
This passage should look like a snippet from an academic paper that perfectly answers the question.
Include specific technical terms, potential numbers, and academic phrasing.
Do NOT answer the question directly, just generate the PASSAGE that would contain the answer.

Examples:
{examples_text}

Question: {question}
Hypothetical Answer Passage:"""

        try:
            hypothetical_passage = await self.llm.complete(prompt, system_prompt="You are a helpful assistant.")
            print(f"--- DEBUG: Query Expansion (Hypothetical Passage) ---\n{hypothetical_passage.strip()[:200]}...\n--------------------------------------------------")
            return f"{question} {hypothetical_passage.strip()}" # 組合原始問題和假設性段落
        except Exception as e:
            print(f"Warning: Query rewrite failed: {e}")
            return question
    
    async def answer(
        self,
        question: str,
        top_k: int = 100, # 增加初篩數量，讓 Reranker 有更多選擇
        llm_top_k: int = 15, # 增加最終上下文數量 (MPNet + Q&A chunk 質量很高)
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None
    ) -> dict:
        """
        回答問題的完整流程
        """
        # 0. 查詢擴展 (Query Expansion)
        search_query = question
        if self.example_retriever:
             search_query = await self.rewrite_query(question)

        # 1. 檢索相關片段 (使用擴展後的查詢) - 獲取大量候選
        results_original = self.vector_store.search(question, top_k=top_k)
        results_expanded = self.vector_store.search(search_query, top_k=top_k)
        
        seen_texts = set()
        search_results = []
        
        for res in results_original + results_expanded:
            if res['text'] not in seen_texts:
                search_results.append(res)
                seen_texts.add(res['text'])
        
        # 2. Reranking (重排序)
        if self.reranker and search_results:
            print(f"--- DEBUG: Reranking {len(search_results)} chunks ---")
            # 準備 (query, text) 對
            pairs = [(question, res['text']) for res in search_results]
            
            try:
                # 預測分數
                scores = self.reranker.predict(pairs)
                
                # 更新分數並排序
                for i, score in enumerate(scores):
                    search_results[i]['rerank_score'] = float(score)
                    
                # 根據 rerank 分數排序
                search_results.sort(key=lambda x: x['rerank_score'], reverse=True)
                
                # 只保留最相關的 llm_top_k 個
                search_results = search_results[:llm_top_k]
                print(f"--- DEBUG: After Reranking, top score: {search_results[0]['rerank_score']:.3f} ---")
            except Exception as e:
                print(f"Warning: Reranking failed, using original order. Error: {e}")
                search_results = search_results[:llm_top_k]
        else:
            # 如果沒有 Reranker，直接截斷
            search_results = search_results[:llm_top_k]
        
        # 調試輸出
        print(f"--- DEBUG: 最終提交給 LLM 的上下文 ({len(search_results)} chunks) ---")
        if search_results:
            print(f"Top 1 Chunk: {search_results[0]['text'][:200]}...")
        print("-" * 40)
        
        # 3. 格式化上下文
        context_parts = []
        ref_ids = []
        
        for i, result in enumerate(search_results):
            doc_id = result['metadata'].get('doc_id', 'unknown')
            page_num = result['metadata'].get('page_num', 0)
            text = result['text']
            # 優先顯示 rerank_score
            score = result.get('rerank_score', result.get('score', 0.0))
            
            context_parts.append(
                f"[{i+1}] Source: {doc_id}, Page: {page_num}, Score: {score:.3f}\n{text}"
            )
            
            if doc_id and doc_id != 'unknown' and doc_id not in ref_ids:
                ref_ids.append(doc_id)
        
        context = "\n\n".join(context_parts) if context_parts else "No relevant context found."
        
        # 4. 獲取示例 (Dynamic or Static)
        examples_text = ""
        if self.example_retriever:
            try:
                examples = self.example_retriever.retrieve(question, k=3)
                if examples:
                    examples_text = self.example_retriever.format_examples(examples)
                    print(f"--- DEBUG: 已檢索並注入 {len(examples)} 個相關示例 ---")
            except Exception as e:
                print(f"Warning: Failed to retrieve examples: {e}")
        
        if not examples_text:
             examples_text = """**Example 1**
Question: "What is the estimated CO2 emissions (in pounds) from training the BERT-base model?"
Answer: {"answer": "1438 lbs", "answer_value": "1438", "answer_unit": "lbs", "supporting_materials": "Table 3", "explanation": "Extracted directly from Table 3", "ref_id": ["strubell2019"]}"""

        # 5. 組裝 prompt (加入 Chain of Thought)
        if user_template is None:
            user_template = """You are an expert research assistant. Answer questions based STRICTLY on the provided context.

CONTEXT:
{context}

QUESTION: {question}

---
### FEW-SHOT EXAMPLES (Similar questions from training data):

{examples}

---

### YOUR TASK:
1. **Analyze**: Read the context carefully and look for keywords from the question.
2. **Reason**: Think step-by-step. Does the context contain the exact answer? Is it explicitly stated?
3. **Extract**: Get the precise value, unit, and verbatim quote.
4. **Format**: Return JSON.

**Critical Rules**:
- If the answer is NOT in the context, return "is_blank" for all fields.
- For True/False: use "1" for True, "0" for False.
- Do NOT use outside knowledge.
- "supporting_materials" MUST be a verbatim quote or "Table X" / "Figure Y".
- "explanation" should be your step-by-step reasoning.

Respond in JSON:
{{
    "answer": "Final natural language answer",
    "answer_value": "The exact extracted value (or 'is_blank')",
    "answer_unit": "The unit of measurement (e.g., 'lbs', 'kWh', or 'is_blank')",
    "supporting_materials": "Verbatim quote from the text proving the answer",
    "explanation": "Reasoning connecting the quote to the answer",
    "ref_id": ["doc_id"]
}}"""
        
        # 替換變量
        user_prompt = user_template.format(
            question=question,
            context=context,
            examples=examples_text
        )
        
        # 6. 呼叫 LLM
        raw_response = await self.llm.complete(
            user_prompt,
            system_prompt=system_prompt or "你是一個有用的助手。"
        )
        
        # 7. 返回結果
        return {
            "question": question,
            "raw_response": raw_response,
            "context": context,
            "ref_ids": ref_ids,
            "search_results": search_results
        }
