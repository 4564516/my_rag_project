"""
Pipeline Step 2: 視覺數據提取 (Vision Extraction)
-----------------------------------------------
功能：
1. 掃描 Markdown 中的圖片引用。
2. 使用 `llama3.2-vision` 模型分析圖片內容。
3. 應用 "Generative Q&A" 策略：生成預測性問答對 (Q&A Pairs) 以最大化檢索命中率。
4. 生成語義摘要 (Context) 用於廣泛搜索。
5. 將生成的描述寫回 Markdown 文件。

執行方式：
python scripts/pipeline_step2_vision.py
"""

import os
import re
import base64
import hashlib
from pathlib import Path
import requests
from tqdm import tqdm

# 配置
OLLAMA_API_URL = "http://localhost:11434/api/generate"
# 優先使用的模型列表 (按順序嘗試)
VISION_MODELS_TO_TRY = ["minicpm-v", "llama3.2-vision"] 
VISION_MODEL = "llama3.2-vision" # 默認值，會被自動切換

PROMPT = """You are an expert AI researcher creating a "Cheat Sheet" for a QA system.
Your goal is to extract every single piece of data from this image and format it as Question-Answer pairs.

**INSTRUCTIONS:**

1.  **Analyze the Image:** Look at every chart, table, legend, and text annotation.
2.  **Generate Q&A Pairs:** For EVERY data point, generate a specific question and its exact answer.
    *   *Example Table Row:* | LLaMA | 65B | 130GB |
    *   *Generate:* 
        *   Q: What is the size of the LLaMA 65B model? A: 130GB.
        *   Q: How many parameters does LLaMA have? A: 65B.

3.  **Cover These Topics Specifically:**
    *   **Carbon Emissions / CO2e:** (e.g., "Q: What were the net CO2e emissions for GShard? A: 4.3 tCO2e")
    *   **Model Sizes / Parameters:** (e.g., "Q: What is the file size of LLaMA-33B? A: 64.7 GB")
    *   **Energy Consumption:** (e.g., "Q: What was the total energy consumption? A: 123 MWh")
    *   **Datasets:** (e.g., "Q: What dataset was used? A: Common Crawl")
    *   **Hardware/GPUs:** (e.g., "Q: How many GPUs were used? A: 128 A100s")
    *   **Performance/Accuracy:** (e.g., "Q: What is the accuracy on ImageNet? A: 85.3%")

4.  **CRITICAL RULE:** 
    *   Try your best to extract ANY text or data visible. If absolutely no data is visible, describe the visual content briefly.
    *   Do NOT hallucinate or repeat "What is the...".

5.  **OUTPUT FORMAT (Strict):**
    
    **Figure Context:**
    [A brief 3-sentence summary of what this image is about, for broad search.]

    **Figure Data (Q&A):**
    Q: [Question 1]? A: [Exact Answer]
    Q: [Question 2]? A: [Exact Answer]
    ...
    (Generate as many as needed to cover ALL data points)

    **Figure Data (Table):**
    [The raw Markdown table for backup]
"""

# 全局緩存：{image_hash: description}
processed_images_cache = {}

def get_image_hash(image_path):
    """計算圖片的 MD5 Hash (指紋)"""
    hash_md5 = hashlib.md5()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_description(image_path):
    try:
        # 編碼圖片（可能需要一些時間）
        print(" [編碼中...]", end="", flush=True)
        b64_image = encode_image(image_path)
        
        # 嘗試每個模型
        for model_name in VISION_MODELS_TO_TRY:
        payload = {
                "model": model_name,
            "prompt": PROMPT,
            "images": [b64_image],
            "stream": False,
            "options": {
                    "num_ctx": 2048,
                    "num_predict": 800,
                    "temperature": 0.0,
                    "repeat_penalty": 1.5,
                    "top_p": 0.9,
                    "top_k": 40
            }
        }
        
            # print(f" [使用模型: {model_name}]", end="", flush=True)
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=(10, 300))
            
        if response.status_code == 200:
                result = response.json().get("response", "")
                # 如果成功獲得結果，直接返回
                if result:
                     # print(f" ✓ {model_name} 成功", flush=True)
                     return result
            else:
                # 如果這個模型失敗 (例如 404 模型不存在)，就嘗試下一個
                # print(f" ✗ {model_name} 失敗 ({response.status_code}), 嘗試下一個...", flush=True)
                continue

        # 如果所有模型都失敗
        print(f" ✗ 所有視覺模型皆失敗", flush=True)
        return ""

    except requests.exceptions.Timeout as e:
        # 超時（連接超時或讀取超時）
        if "ConnectTimeout" in str(type(e)) or "連接" in str(e):
            print(f" ✗ 連接超時：無法連接到 Ollama 服務", flush=True)
            print(f"   請確認 Ollama 服務正在運行：ollama serve", flush=True)
        else:
            print(f" ✗ 處理超時：單張圖片處理超過 5 分鐘，可能卡住了", flush=True)
            print(f"   建議：跳過此圖片或檢查圖片大小", flush=True)
        return ""
    except requests.exceptions.ConnectionError as e:
        # 連接被拒絕或無法建立連接
        print(f" ✗ 連接錯誤：無法連接到 Ollama 服務 (http://localhost:11434)", flush=True)
        print(f"   錯誤詳情: {str(e)[:150]}", flush=True)
        print(f"   請確認：", flush=True)
        print(f"   1. Ollama 服務正在運行：ollama serve", flush=True)
        print(f"   2. 服務監聽在正確的端口 (11434)", flush=True)
            return ""
    except Exception as e:
        print(f" ✗ 處理失敗: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return ""

def clean_markdown_content(content):
    """
    智能清除舊的 Figure Description，並清理重複內容和大量空白行。
    邏輯：遇到 **Figure Description:** 開始刪除，直到遇到可能的正文/標題/新圖片/表格。
    """
    lines = content.split('\n')
    new_lines = []
    skipping = False
    last_line = None
    repeat_count = 0
    
    for i, line in enumerate(lines):
        # 檢測重複行（連續相同的行超過3次，可能是模型輸出錯誤）
        if line.strip() == last_line and line.strip():
            repeat_count += 1
            if repeat_count > 3:
                # 跳過重複行
                continue
        else:
            repeat_count = 0
            last_line = line.strip() if line.strip() else None
        
        # 檢查是否開始進入需要清理的區塊
        # 我們需要清理：
        # 1. 舊版：**Figure Description:**
        # 2. 新版：**Figure Context:** 和 **Figure Data:**
        # 3. 激進版：# Table Processing, # Chart/Plot Processing, **Chart/PLOT**
        # 4. 垃圾版："Q: What is the" 連續出現
        
        lower_line = line.lower()
        if ("**figure context:**" in lower_line or 
            "**figure data:**" in lower_line or
            "**figure description:**" in lower_line or
            "# table processing" in lower_line or
            "# chart/plot processing" in lower_line or
            "# chart/plot processing" in lower_line or
            "**chart/plot**" in lower_line or
            line.strip().startswith("Q: What is the") or # 清理殘留的垃圾問題
            line.strip().startswith("Q: What is a") # 清理殘留的垃圾問題
           ):
            skipping = True
            continue
            
        if skipping:
            # 決定何時停止刪除
            # 條件 1: 遇到新的圖片引用
            if line.strip().startswith('!['):
                skipping = False
            # 條件 2: 遇到標題 (但排除我們要刪除的特定標題)
            elif line.strip().startswith('#'):
                # 再次檢查這是不是我們要刪除的標題
                if not ("table processing" in lower_line or "chart/plot" in lower_line):
                skipping = False
            # 條件 3: 遇到明顯的 Figure Caption
            elif line.strip().startswith('Figure ') or line.strip().startswith('Table '):
                skipping = False
            # 條件 4: 遇到引用標記
            elif '<span id=' in line:
                skipping = False
            
            # 條件 5: 遇到空行後跟隨非空行（可能是新段落）
            elif not line.strip() and i + 1 < len(lines) and lines[i + 1].strip():
                next_line = lines[i + 1].strip()
                next_lower = next_line.lower()
                # 檢查下一行是否是新的內容
                if (not next_line.startswith('*') and 
                    not next_line.startswith('-') and
                    not "figure context:" in next_lower and
                    not "figure data:" in next_lower and
                    not "figure description:" in next_lower and
                    not "table processing" in next_lower and
                    not "chart/plot" in next_lower and
                    not next_line.startswith('|')):
                skipping = False
            
            if not skipping:
                new_lines.append(line)
            # else: continue skipping (delete this line)
        else:
            new_lines.append(line)
            
    # 再次清理：移除連續重複的短行（可能是模型輸出錯誤）
    cleaned_lines = []
    for i, line in enumerate(new_lines):
        # 跳過非常短的重複行（可能是模型輸出錯誤）
        if len(line.strip()) < 20 and i > 0 and line.strip() == new_lines[i-1].strip() and line.strip():
            continue
        cleaned_lines.append(line)
    
    # 清理大量連續空白行（超過3行的空白行，只保留最多2行）
    final_lines = []
    consecutive_empty = 0
    
    for line in cleaned_lines:
        if not line.strip():
            consecutive_empty += 1
            if consecutive_empty <= 2:  # 最多保留2行空白
                final_lines.append(line)
            # 超過2行的空白行會被跳過
        else:
            consecutive_empty = 0
            final_lines.append(line)
            
    return '\n'.join(final_lines)

def check_ollama_service():
    """檢查 Ollama 服務是否運行"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            # 檢查模型是否存在（支援帶或不帶 :latest 後綴）
            # 例如：llama3.2-vision 可以匹配 llama3.2-vision:latest
            model_base = VISION_MODEL.split(":")[0]  # 去掉可能的後綴
            found_model = None
            for name in model_names:
                if name.startswith(model_base + ":") or name == model_base:
                    found_model = name
                    break
            
            if found_model:
                print(f"✅ Ollama 服務運行中，已找到模型: {found_model}")
                return True
            else:
                print(f"⚠️  Ollama 服務運行中，但未找到模型: {VISION_MODEL}")
                print(f"   可用模型: {', '.join(model_names) if model_names else '無'}")
                print(f"   請執行: ollama pull {VISION_MODEL}")
                return False
        else:
            print(f"⚠️  Ollama 服務響應異常 (狀態碼 {response.status_code})")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ 無法連接到 Ollama 服務 (http://localhost:11434)")
        print(f"   請確認 Ollama 已安裝並運行：ollama serve")
        return False
    except Exception as e:
        print(f"⚠️  檢查 Ollama 服務時發生錯誤: {e}")
        return False

def process_docs_with_vision():
    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "artifacts/processed_docs"
    
    if not docs_dir.exists():
        print("❌ No processed docs found. Please run step 1 first.")
        return

    # 檢查 Ollama 服務
    print("檢查 Ollama 服務狀態...")
    if not check_ollama_service():
        print("\n⚠️  警告：Ollama 服務檢查失敗，但將繼續嘗試處理...")
        print("   如果遇到連接錯誤，請先啟動 Ollama 服務\n")

    print(f"Searching for Markdown files in {docs_dir}...")
    md_files = list(docs_dir.rglob("*.md"))
    
    if not md_files:
        print("❌ No Markdown files found.")
        return

    print(f"Found {len(md_files)} Markdown files.\n")

    for md_file in tqdm(md_files, desc="Processing Docs"):
        doc_dir = md_file.parent 
        content = md_file.read_text(encoding="utf-8")
        
        # 1. 清理所有舊描述
        clean_content = clean_markdown_content(content)
        
        # 2. 掃描圖片並生成新描述
        images_found = list(re.finditer(r'!\[(.*?)\]\((.*?)\)', clean_content))
        
        if not images_found:
            # 如果清理後沒圖片了（不應該發生），或者本來就沒圖片
            if clean_content != content:
                 md_file.write_text(clean_content, encoding="utf-8") # 至少保存清理後的結果
            continue
            
        print(f"  Found {len(images_found)} images in {md_file.name}")
        
        # 3. 處理所有圖片
        final_content_parts = []
        last_pos = 0
        updated = False
        
        for match in images_found:
            start, end = match.span()
            # 把上一個圖片到這一個圖片之間的文字加進去
            final_content_parts.append(clean_content[last_pos:end]) 
            last_pos = end # 移動游標到圖片引用結束處 (![...](...))
            
            # 處理圖片
            full_match_str = match.group(0)
            img_rel_path = match.group(2)
            
            if img_rel_path.startswith("http"):
                continue # 保持原樣 (已經 append 了)

            img_full_path = doc_dir / img_rel_path
            
            # URL decode logic
            if not img_full_path.exists():
                 try:
                    import urllib.parse
                    decoded_path = urllib.parse.unquote(img_rel_path)
                    img_full_path = doc_dir / decoded_path
                 except:
                     pass

            if not img_full_path.exists():
                continue

            # --- 圖片去重邏輯 (Image Deduplication) ---
            img_hash = get_image_hash(img_full_path)
            if img_hash in processed_images_cache:
                print(f"    [Cache Hit] Skipping duplicate image {img_rel_path}...", flush=True)
                description = processed_images_cache[img_hash]
            else:
                print(f"    Captioning {img_rel_path}...", end="", flush=True)
            description = get_image_description(img_full_path)
                
                # 只有當描述有效時才快取 (避免快取錯誤)
                # 移除 NO_DATA_EXTRACTED 檢查，確保 cache 住所有結果（即使是空數據，避免重複跑空數據）
                if description:
                     processed_images_cache[img_hash] = description
            
            if description:
                # 如果是 "NO_DATA_EXTRACTED"，我們就直接跳過不寫入
                # if "NO_DATA_EXTRACTED" in description:
                #      # print(f"      [Skipped] No data extracted from {img_rel_path}", flush=True)
                #      continue # 跳過，不寫入描述到 Markdown
                # 清理描述中的重複內容
                original_length = len(description)
                
                # 第一階段：清理重複行
                description_lines = description.split('\n')
                cleaned_description_lines = []
                last_desc_line = None
                repeat_count = 0
                
                for desc_line in description_lines:
                    # 移除連續重複的行
                    if desc_line.strip() == last_desc_line and desc_line.strip():
                        repeat_count += 1
                        if repeat_count > 1:  # 只允許最多1次重複
                            continue
                    else:
                        repeat_count = 0
                        last_desc_line = desc_line.strip() if desc_line.strip() else None
                    
                    # 移除過短的重複行（可能是模型輸出錯誤）
                    if len(desc_line.strip()) < 10 and desc_line.strip() == last_desc_line:
                        continue
                        
                    cleaned_description_lines.append(desc_line)
                
                cleaned_description = '\n'.join(cleaned_description_lines)
                
                # 第二階段：檢測並移除重複的段落/區塊（更積極的清理）
                if len(cleaned_description) > 500:  # 進一步降低閾值，更早開始清理
                    # 按段落分割
                    paragraphs = cleaned_description.split('\n\n')
                    seen_prefixes = set()  # 存儲段落前綴
                    seen_hashes = set()    # 存儲段落 hash
                    unique_paragraphs = []
                    
                    for para in paragraphs:
                        para_stripped = para.strip()
                        if not para_stripped:
                            continue
                        
                        # 計算段落的簡化版本（用於檢測重複）
                        # 使用更長的比較長度（200字元）來檢測相似段落
                        para_simple = para_stripped[:200] if len(para_stripped) > 200 else para_stripped
                        para_hash = hash(para_stripped)  # 使用完整段落的 hash 來檢測完全重複
                        
                        # 如果段落開頭相似（前200字元）或完全重複，跳過
                        if para_simple not in seen_prefixes and para_hash not in seen_hashes:
                            seen_prefixes.add(para_simple)
                            seen_hashes.add(para_hash)
                            unique_paragraphs.append(para)
                        # 如果段落重複，跳過
                    
                    cleaned_description = '\n\n'.join(unique_paragraphs)
                
                # 第三階段：強制限制最終長度（最多 1500 字元）
                # 無論如何都要截斷，確保不會超過限制
                if len(cleaned_description) > 1500:
                    # 截斷到 1500 字元，但嘗試在句子邊界截斷
                    truncated = cleaned_description[:1500]
                    # 找到最後一個句號或換行
                    last_period = truncated.rfind('.')
                    last_newline = truncated.rfind('\n')
                    cut_point = max(last_period, last_newline)
                    if cut_point > 1000:  # 確保不會截斷太多
                        cleaned_description = truncated[:cut_point+1] + "\n\n[描述已截斷以避免過長]"
                    else:
                        cleaned_description = truncated + "\n\n[描述已截斷以避免過長]"
                
                # 報告清理效果
                if original_length > len(cleaned_description):
                    reduction = original_length - len(cleaned_description)
                    reduction_pct = (reduction / original_length) * 100
                    print(f" (清理: {original_length:,} → {len(cleaned_description):,} 字元, 減少 {reduction_pct:.1f}%)", flush=True)
                
                final_content_parts.append(f"\n\n**Figure Description:**\n{cleaned_description}\n\n")
                updated = True
            else:
                print(f"      ⚠️  未獲得描述，跳過", flush=True)
        
        # 把剩下的一段加進去
        final_content_parts.append(clean_content[last_pos:])
        
        new_full_content = "".join(final_content_parts)
        
        if updated or clean_content != content:
            md_file.write_text(new_full_content, encoding="utf-8")
            print(f"  ✅ Updated {md_file.name}")

    print("\n✅ Step 2 Complete! Images are captioned in Markdown.")

if __name__ == "__main__":
    process_docs_with_vision()
