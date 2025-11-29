"""
答案格式化模組 (Answer Formatter)
-----------------------------
功能：
1. 解析 LLM 的 JSON 輸出。
2. 進行數據清洗與標準化 (單位轉換、布林值處理)。
3. 將數據轉換為符合提交格式的 CSV 行。
4. 處理 is_blank 或格式錯誤的邊界情況。
"""

import json
import re
import csv
from pathlib import Path
from typing import Optional, List, Dict, Any

class AnswerFormatter:
    """用於格式化 LLM 輸出的答案，以匹配競賽的 CSV 欄位"""
    
    BLANK_TOKEN = "is_blank"
    UNANSWERABLE_PHRASE = "Unable to answer with confidence based on the provided documents."

    def __init__(self, metadata_path: Optional[str] = None):
        self.metadata_map = {}
        if metadata_path:
            self.load_metadata(metadata_path)

    def load_metadata(self, path: str):
        """加載 metadata.csv 以建立 id -> url 映射"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "id" in row and "url" in row:
                        self.metadata_map[row["id"].strip()] = row["url"].strip()
        except Exception as e:
            print(f"Warning: Failed to load metadata from {path}: {e}")

    def parse_llm_json(self, raw_response: str) -> Dict[str, Any]:
        """嘗試從原始回應中解析 JSON，包含錯誤恢復機制"""
        if not raw_response:
            return {}
            
        # 1. 基本清理
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        # 2. 嘗試標準 JSON 解析
        match = re.search(r'\{[\s\S]*\}', cleaned)
        if match:
            json_str = match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # 3. 常見錯誤修復
                try:
                    json_str_fixed = re.sub(r',\s*\}', '}', json_str)
                    return json.loads(json_str_fixed)
                except json.JSONDecodeError:
                    pass
                    
        # 4. 暴力提取
        print(f"DEBUG: JSON 解析失敗，嘗試暴力提取: {cleaned[:100]}...")
        result = {}
        
        # 提取 answer_value
        val_match = re.search(r'"answer_value"\s*:\s*"([^"]*)"', cleaned)
        if val_match: result["answer_value"] = val_match.group(1)
        
        # 提取 answer
        ans_match = re.search(r'"answer"\s*:\s*"([^"]*)"', cleaned)
        if ans_match: result["answer"] = ans_match.group(1)

        # 提取 answer_unit
        unit_match = re.search(r'"answer_unit"\s*:\s*"([^"]*)"', cleaned)
        if unit_match: result["answer_unit"] = unit_match.group(1)

        # 提取 supporting_materials
        supp_match = re.search(r'"supporting_materials"\s*:\s*"([^"]*)"', cleaned)
        if supp_match: result["supporting_materials"] = supp_match.group(1)
            
        # 提取 ref_id
        ref_match = re.search(r'"ref_id"\s*:\s*\[(.*?)\]', cleaned)
        if ref_match:
            ids = [x.strip().strip('"').strip("'") for x in ref_match.group(1).split(',')]
            result["ref_id"] = [i for i in ids if i]
            
        return result

    def normalize_answer_value(self, raw_value: Optional[Any], question: str) -> str:
        """正規化答案值"""
        value = str(raw_value or "").strip() 
        
        if not value or value.lower() == self.BLANK_TOKEN.lower() or value.lower() == self.UNANSWERABLE_PHRASE.lower():
            return self.BLANK_TOKEN
        
        # 處理 True/False -> 1/0
        # 先檢查是否為 True/False 問題
        is_boolean_question = question.lower().strip().startswith("true or false") or question.lower().strip().startswith("true/false")

        # 1. 明確的文字映射 (無論題目類型)
        if str(value).lower() in ["true", "yes", "1", "correct"]:
            return "1"
        if str(value).lower() in ["false", "no", "0", "incorrect"]:
            return "0"
            
        # 2. 如果是 True/False 問題，進行更激進的過濾
        if is_boolean_question:
            # 如果值看起來不像 0 或 1，嘗試從更長的文本中提取
            # 比如 value 是 "The statement is False because..." -> 提取 "False" -> "0"
            if "false" in str(value).lower() or "incorrect" in str(value).lower():
                return "0"
            if "true" in str(value).lower() or "correct" in str(value).lower():
                return "1"
            
            # 如果無法識別為 True/False，且不是明確的 is_blank，則強制轉為 is_blank
            # 這是為了防止出現 "19474" 這種奇怪的數字
            if value != self.BLANK_TOKEN:
                print(f"Warning: Invalid boolean value '{value}' for True/False question. Forcing to is_blank.")
                return self.BLANK_TOKEN

        if value.startswith('[') and value.endswith(']'):
            return value # Assume correct format if brackets exist

        # 處理 million/billion/trillion
        value_lower = value.lower()
        multiplier = 1
        if "trillion" in value_lower:
            multiplier = 1_000_000_000_000
        elif "billion" in value_lower:
            multiplier = 1_000_000_000
        elif "million" in value_lower:
            multiplier = 1_000_000
            
        if multiplier > 1:
            num_match = re.search(r'[\d,\.]+', value)
            if num_match:
                try:
                    num = float(num_match.group(0).replace(',', ''))
                    return str(int(num * multiplier))
                except ValueError:
                    pass
                
        # 簡單數字提取
        has_letters = bool(re.search(r'[a-zA-Z]', value))
        # 允許 "days", "years" 等單位詞，或 "~", "approx" 等修飾詞
        ignore_words = ["days", "day", "years", "year", "samples", "liters", "approx", "approximately", "about", "~"]
        clean_val_for_check = value.lower()
        for w in ignore_words:
            clean_val_for_check = clean_val_for_check.replace(w, "")
            
        has_letters_after_clean = bool(re.search(r'[a-zA-Z]', clean_val_for_check))
        
        num_match = None
        if not has_letters_after_clean: # 如果清理後只剩數字和符號
            num_match = re.search(r'[\d,\.]+', value)
        
        if num_match:
            return num_match.group(0).replace(',', '')
            
        return value

    def format_answer(
        self,
        raw_response: str,
        retrieved_ref_ids: List[str],
        question: str
    ) -> Dict[str, str]:
        """
        將 RAG Pipeline 的結果格式化為最終輸出
        """
        llm_output = self.parse_llm_json(raw_response)
        
        answer = llm_output.get("answer", self.UNANSWERABLE_PHRASE)
        answer_value_raw = llm_output.get("answer_value", self.BLANK_TOKEN)
        answer_unit = llm_output.get("answer_unit", self.BLANK_TOKEN)
        supporting_materials = llm_output.get("supporting_materials", "")
        explanation = llm_output.get("explanation", "")
        llm_ref_ids = llm_output.get("ref_id", [])

        normalized_value = self.normalize_answer_value(answer_value_raw, question)
        
        # 決定最終狀態
        is_unanswerable = (
            normalized_value == self.BLANK_TOKEN or 
            answer.lower() == self.UNANSWERABLE_PHRASE.lower() or
            "unable to answer" in answer.lower()
        )

        if is_unanswerable:
            final_answer = self.UNANSWERABLE_PHRASE
            final_value = self.BLANK_TOKEN
            final_unit = self.BLANK_TOKEN
            final_ref_id = self.BLANK_TOKEN
            final_ref_url = self.BLANK_TOKEN
            final_supporting = self.BLANK_TOKEN
            final_explanation = self.BLANK_TOKEN
        else:
            final_answer = answer
            final_value = normalized_value
            final_unit = answer_unit
            final_supporting = supporting_materials
            final_explanation = explanation

            # 處理 Ref IDs
            if llm_ref_ids and isinstance(llm_ref_ids, list) and len(llm_ref_ids) > 0:
                valid_ref_ids = [rid for rid in llm_ref_ids if rid and rid != 'unknown' and rid != 'is_blank']
                # Jaccard overlap 需要 ref_id 是列表格式的字符串，如 "['doc1', 'doc2']"
                # 但 csv writer 會自動處理列表，這裡我們轉為字符串以符合格式要求
                final_ref_id_list = valid_ref_ids
            else:
                final_ref_id_list = retrieved_ref_ids if retrieved_ref_ids else []
            
            if not final_ref_id_list:
                final_ref_id = self.BLANK_TOKEN
                final_ref_url = self.BLANK_TOKEN
            else:
                # 格式化為 ['id1', 'id2']
                final_ref_id = str(final_ref_id_list).replace('"', "'")
                
                # 查找 URLs
                urls = []
                for rid in final_ref_id_list:
                    url = self.metadata_map.get(rid, "")
                    if url:
                        urls.append(url)
                
                if urls:
                    final_ref_url = str(urls).replace('"', "'")
                else:
                    final_ref_url = self.BLANK_TOKEN

        return {
            "answer": final_answer,
            "answer_value": final_value,
            "answer_unit": final_unit,
            "ref_id": final_ref_id, 
            "ref_url": final_ref_url,
            "supporting_materials": final_supporting,
            "explanation": final_explanation
        }
