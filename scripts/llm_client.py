"""
步驟 5: LLM 客戶端模組

功能：
- 與 OpenAI/OpenRouter API 互動
- 發送請求並處理回應
- 錯誤處理和重試
"""

import os
import asyncio
from typing import Optional
# 確保您已經安裝了 openai 庫： pip install openai
from openai import AsyncOpenAI


class LLMClient:
    """LLM API 客戶端"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        # 將預設模型修改為 Mistral 7B Instruct
        model: str = "mistralai/mistral-7b-instruct:free" 
    ):
        """
        初始化 LLM 客戶端
        
        Args:
            api_key: API Key（如果為 None，會從環境變數讀取）
            base_url: API Base URL（OpenRouter 需要）
            model: 模型名稱
        """
        # 從環境變數或參數取得 API Key
        # 檢查是否使用 Ollama（本地模型，不需要真實的 API key）
        model_lower = model.lower()
        is_ollama = "ollama" in model_lower or model.startswith("ollama/")
        
        if is_ollama:
            # Ollama 不需要真實的 API key，但需要設置一個值
            self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "ollama"
            print("DEBUG: 使用 Ollama 本地模型（無需 API key）")
        else:
            # 其他 API 需要真實的 key
            self.api_key = (api_key or 
                          os.getenv("OPENROUTER_API_KEY") or 
                          os.getenv("GROQ_API_KEY") or
                          os.getenv("HUGGINGFACE_API_KEY") or
                          os.getenv("OPENAI_API_KEY"))
            
        if not self.api_key:
            raise ValueError(
                "需要提供 API Key！\n\n"
                "免費選項：\n"
                "  1. 使用 Ollama（本地模型，完全免費）：\n"
                "     - 安裝: brew install ollama\n"
                "     - 下載模型: ollama pull llama3.2:3b\n"
                "     - 設置: export LLM_MODEL='ollama/llama3.2:3b'\n"
                "     - 設置: export OPENAI_BASE_URL='http://localhost:11434/v1'\n"
                "     - 設置: export OPENAI_API_KEY='ollama'\n\n"
                "  2. OpenRouter（每天 50 次免費）：\n"
                "     - 獲取 key: https://openrouter.ai/keys\n"
                "     - 設置: export OPENROUTER_API_KEY='your-key'\n\n"
                "  3. Groq（免費，速度快）：\n"
                "     - 獲取 key: https://console.groq.com/keys\n"
                "     - 設置: export GROQ_API_KEY='your-key'\n"
                "     - 設置: export LLM_MODEL='llama-3.1-8b-instant'\n\n"
                "詳見: FREE_API_OPTIONS.md"
            )
        
        # 設定 base_url
        # 優先使用環境變數或參數中的 base_url
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        
        # 如果沒有設定 base_url，根據模型名稱自動判斷
        if not self.base_url:
            model_lower = model.lower()
            if "ollama" in model_lower or model.startswith("ollama/"):
                # Ollama 本地模型
                self.base_url = "http://localhost:11434/v1"
                print(f"DEBUG: 使用 Ollama 本地模型，Base URL: {self.base_url}")
            elif "groq" in model_lower:
                # Groq API
                self.base_url = "https://api.groq.com/openai/v1"
                print(f"DEBUG: 使用 Groq API，Base URL: {self.base_url}")
            elif "openrouter" in model_lower or "mistralai" in model_lower or "/" in model:
                # OpenRouter API
                self.base_url = "https://openrouter.ai/api/v1"
                print(f"DEBUG: 使用 OpenRouter API，Base URL: {self.base_url}")
            else:
                # 默認使用 OpenAI API
                self.base_url = "https://api.openai.com/v1"
                print(f"DEBUG: 使用 OpenAI API，Base URL: {self.base_url}")
        
        # 處理模型名稱（Ollama 需要去掉前綴）
        if is_ollama and model.startswith("ollama/"):
            self.model = model.replace("ollama/", "")
        else:
            self.model = model
        
        # 建立客戶端
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3
    ) -> str:
        """
        發送請求並取得回答
        
        Args:
            prompt: 使用者 prompt
            system_prompt: 系統 prompt（可選）
            temperature: 溫度參數
            max_retries: 最大重試次數
        
        Returns:
            LLM 的回答文字
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # 簡單的重試邏輯
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                error_str = str(e)
                # 檢查是否是速率限制錯誤
                if "429" in error_str or "Rate limit" in error_str or "rate limit" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = 5 * (attempt + 1)  # 速率限制時等待更長時間
                        print(f"⚠️  API 速率限制，等待 {wait_time} 秒後重試...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"❌ API 速率限制，已達最大重試次數。")
                        print(f"   提示：免費模型每天有 50 次請求限制。")
                        print(f"   建議：減少問題數量或使用付費 API key。")
                        raise RuntimeError(f"API 速率限制：{error_str}")
                elif attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指數退避
                    print(f"請求失敗，{wait_time} 秒後重試... ({e})")
                    await asyncio.sleep(wait_time)
                else:
                    raise


# 測試程式碼
if __name__ == "__main__":
    async def test():
        # 測試 LLM 客戶端，使用 Mistral 7B Instruct (OpenRouter 免費模型)
        # 注意：如果您沒有傳入 model 參數，它現在會使用類別定義的預設值
        client = LLMClient(model="mistralai/mistral-7b-instruct:free")
        
        response = await client.complete(
            prompt="什麼是 RAG？用一句話回答。",
            system_prompt="你是一個有用的助手。"
        )
        print(f"回答: {response}")
    
    asyncio.run(test())