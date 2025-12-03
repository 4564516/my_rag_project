"""
Pipeline Step 1: PDF 解析 (Marker)
--------------------------------
功能：
1. 使用 `marker-pdf` 工具將原始 PDF 轉換為高品質 Markdown。
2. 提取 PDF 中的所有圖片並保存到 `processed_docs`。
3. 保存元數據 (Metadata JSON)。

執行方式：
python scripts/pipeline_step1_marker.py
"""

import os
import subprocess
from pathlib import Path
from tqdm import tqdm

def convert_pdfs_with_marker():
    # 設定路徑
    project_root = Path(__file__).parent.parent
    pdf_dir = project_root / "artifacts/raw_pdfs"
    output_base = project_root / "artifacts/processed_docs"
    
    output_base.mkdir(parents=True, exist_ok=True)
    
    # 獲取所有 PDF
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs to process.")
    
    # 檢查 marker 命令是否可用
    try:
        subprocess.run(["marker_single", "--help"], capture_output=True)
    except FileNotFoundError:
        print("❌ Error: 'marker_single' command not found.")
        print("Please ensure marker-pdf is installed: pip install marker-pdf")
        print("And you might need to install system dependencies (like tesseract).")
        return

    # 設定環境變數
    env = os.environ.copy()
    # 移除強制 CUDA 設定，讓它自動偵測 (因為檢測到目前 PyTorch 僅支援 CPU)
    # env["TORCH_DEVICE"] = "cuda" 
    # env["SURYA_DEVICE"] = "cuda"

    for pdf_file in tqdm(pdf_files, desc="Converting PDFs"):
        doc_id = pdf_file.stem
        output_dir = output_base / doc_id
        
        # 檢查是否已經轉換過 (檢查是否有 .md 文件)
        if output_dir.exists() and list(output_dir.glob("*.md")):
            print(f"Skipping {doc_id}, already converted.")
            continue
            
        print(f"\nProcessing {doc_id}...")
        
        # 嘗試最傳統的格式: marker_single INPUT OUTPUT
        # 如果這也失敗，我們需要先檢查 marker 版本
        
        cmd = [
            "marker_single",
            str(pdf_file),
            "--output_dir",   # 加入這個旗
            str(output_dir)
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                #capture_output=True, 
                text=True,
                env=env # 使用包含 CUDA 設定的環境變量
            )
            # print(result.stdout) # 調試用
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to convert {doc_id}:")
            print(e.stderr)
            continue
            
    print("\n✅ Step 1 Complete! Markdown files and images are in artifacts/processed_docs/")

if __name__ == "__main__":
    convert_pdfs_with_marker()

