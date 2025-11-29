"""
Dynamic Few-Shot Example Retriever
用于从训练集中检索最相似的问答对，作为 Prompt 的上下文示例
"""
import csv
from typing import List, Dict
from pathlib import Path
import numpy as np
from .embedder import Embedder

class ExampleRetriever:
    def __init__(self, train_csv_path: str, embedder: Embedder):
        self.examples = []
        self.embeddings = None
        self.embedder = embedder
        self.load_examples(train_csv_path)
        
    def load_examples(self, csv_path: str):
        """加载训练集问答对并生成向量"""
        path = Path(csv_path)
        if not path.exists():
            print(f"Warning: Training data not found at {csv_path}")
            return

        print(f"Loading examples from {csv_path}...")
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("question", "").strip() and row.get("answer_value", "").strip():
                    # 只保留有明确答案的样本
                    if row["answer_value"] != "is_blank":
                        self.examples.append({
                            "question": row["question"],
                            "answer": row["answer"],
                            "answer_value": row["answer_value"],
                            "ref_id": row["ref_id"],
                            "explanation": row.get("explanation", "")
                        })
        
        if self.examples:
            # 批量生成 embeddings
            questions = [ex["question"] for ex in self.examples]
            self.embeddings = self.embedder.encode(questions)
            print(f"Loaded and embedded {len(self.examples)} examples.")
            
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """检索与 query 最相似的 k 个例子"""
        if not self.examples or self.embeddings is None:
            return []
            
        # 生成 query 向量
        query_embedding = self.embedder.encode([query])[0]
        
        # 计算余弦相似度
        # embeddings shape: (N, D), query shape: (D,)
        # scores shape: (N,)
        scores = np.dot(self.embeddings, query_embedding)
        
        # 归一化 (假设 embedder 已经输出了归一化向量，如果没有则需要: / (norm(a)*norm(b)))
        # sentence-transformers 默认输出通常未归一化，但这里简化处理，只求 top-k 排序
        
        # 获取 top-k 索引
        top_indices = np.argsort(scores)[::-1][:k]
        
        return [self.examples[i] for i in top_indices]

    def format_examples(self, examples: List[Dict]) -> str:
        """将例子格式化为 Prompt 字符串"""
        if not examples:
            return ""
            
        formatted = []
        for i, ex in enumerate(examples, 1):
            formatted.append(f"**Example {i}**")
            formatted.append(f"Question: \"{ex['question']}\"")
            formatted.append(f"Answer Object: {{\"answer_value\": \"{ex['answer_value']}\", \"answer\": \"{ex['answer']}\", \"ref_id\": {ex['ref_id']}}}")
            formatted.append("")
            
        return "\n".join(formatted)

