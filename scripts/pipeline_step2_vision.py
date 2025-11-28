"""
Pipeline Step 2: Image Captioning with Vision Model (Ollama)

功能：
- 遍歷 artifacts/processed_docs 下的所有圖片
- 使用 Ollama (llama3.2-vision) 生成詳細描述
- 將描述插入回 Markdown 文件
- 強制更新模式：智能清除舊的 Figure Description
"""

import os
import re
import base64
from pathlib import Path
import requests
from tqdm import tqdm

# 配置
OLLAMA_API_URL = "http://localhost:11434/api/generate"
VISION_MODEL = "llama3.2-vision"
PROMPT = """You are a scientific data extractor. Your task is to extract information from this image with high precision.

1. IF THIS IS A TABLE: Transcribe it into a Markdown table format. Content MUST be exact.
2. IF THIS IS A CHART/PLOT: List every visible data point as "Label: Value". Summarize the X and Y axis units.
3. IF THIS IS A DIAGRAM: Describe the flow and labels.

CRITICAL:
- Do not summarize large numbers (e.g. write "1,024" not "1k").
- Extract specific numbers, percentages, and units exactly as shown.
- If there is a legend, describe what each color/symbol represents.
"""

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_description(image_path):
    try:
        b64_image = encode_image(image_path)
        
        payload = {
            "model": VISION_MODEL,
            "prompt": PROMPT,
            "images": [b64_image],
            "stream": False,
            "options": {
                "num_ctx": 2048,  # 限制 context 大小以節省記憶體
                "temperature": 0.1 # 降低隨機性，減少亂碼
            }
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload)
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            print(f"Error from Ollama: {response.text}")
            return ""
    except Exception as e:
        print(f"Failed to process image {image_path}: {e}")
        return ""

def clean_markdown_content(content):
    """
    智能清除舊的 Figure Description。
    邏輯：遇到 **Figure Description:** 開始刪除，直到遇到可能的正文/標題/新圖片。
    """
    lines = content.split('\n')
    new_lines = []
    skipping = False
    
    for i, line in enumerate(lines):
        if "**Figure Description:**" in line:
            skipping = True
            continue
            
        if skipping:
            # 決定何時停止刪除
            # 條件 1: 遇到新的圖片引用
            if line.strip().startswith('!['):
                skipping = False
            # 條件 2: 遇到標題
            elif line.strip().startswith('#'):
                skipping = False
            # 條件 3: 遇到明顯的 Figure Caption (Marker 通常會保留 Figure X: ...)
            elif line.strip().startswith('Figure ') or line.strip().startswith('Table '):
                skipping = False
            # 條件 4: 遇到引用標記 (e.g. <span id=...) - 這通常是新段落的開始
            elif '<span id=' in line:
                skipping = False
            
            if not skipping:
                new_lines.append(line)
            # else: continue skipping (delete this line)
        else:
            new_lines.append(line)
            
    return '\n'.join(new_lines)

def process_docs_with_vision():
    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "artifacts/processed_docs"
    
    if not docs_dir.exists():
        print("❌ No processed docs found. Please run step 1 first.")
        return

    print(f"Searching for Markdown files in {docs_dir}...")
    md_files = list(docs_dir.rglob("*.md"))
    
    if not md_files:
        print("❌ No Markdown files found.")
        return

    print(f"Found {len(md_files)} Markdown files.")

    for md_file in tqdm(md_files, desc="Processing Docs"):
        doc_dir = md_file.parent 
        content = md_file.read_text(encoding="utf-8")
        
        # 1. 清理舊描述
        clean_content = clean_markdown_content(content)
        
        # 2. 掃描圖片並生成新描述
        # 使用 finditer 獲取所有圖片
        images_found = list(re.finditer(r'!\[(.*?)\]\((.*?)\)', clean_content))
        
        if not images_found:
            # 如果清理後沒圖片了（不應該發生），或者本來就沒圖片
            if clean_content != content:
                 md_file.write_text(clean_content, encoding="utf-8") # 至少保存清理後的結果
            continue
            
        print(f"  Found {len(images_found)} images in {md_file.name}")
        
        # 我們需要構建最終內容。為了避免多次 replace 造成的混亂（特別是如果有多個相同圖片引用），
        # 我們這次採取 "Split & Rebuild" 的策略，或者更簡單的：
        # 使用一個游標，從頭到尾構建新字符串。
        
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
                # print(f"Warning: Image not found {img_full_path}")
                continue

            print(f"    Captioning {img_rel_path}...")
            description = get_image_description(img_full_path)
            
            if description:
                final_content_parts.append(f"\n\n**Figure Description:**\n{description}\n\n")
                updated = True
        
        # 把剩下的一段加進去
        final_content_parts.append(clean_content[last_pos:])
        
        new_full_content = "".join(final_content_parts)
        
        if updated or clean_content != content:
            md_file.write_text(new_full_content, encoding="utf-8")
            print(f"  ✅ Updated {md_file.name}")

    print("\n✅ Step 2 Complete! Images are captioned in Markdown.")

if __name__ == "__main__":
    process_docs_with_vision()
