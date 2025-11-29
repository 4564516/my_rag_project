# AI Research Assistant RAG System

這是一個針對學術論文的高精度 RAG (Retrieval-Augmented Generation) 系統。它結合了 SOTA 等級的開源模型與策略，旨在解決科學文獻中複雜表格數據提取與跨文檔檢索的難題。

## 🚀 核心架構 (Architecture)

本系統採用了 **"Heavy Artillery"** 配置，最大化利用本地與開源資源：

### 1. 視覺處理 (Vision Processing)
*   **模型**: `llama3.2-vision` (via Ollama)
*   **策略**: **Generative Q&A (生成式問答)**
    *   不同於傳統的表格轉錄，我們讓視覺模型直接生成「預測考題」(Predictive Q&A)。
    *   例如：`Q: What is the GShard emission? A: 4.3 tCO2e`
    *   這使得數據點能被向量檢索精確命中。

### 2. 檢索系統 (Retrieval System)
*   **Embedding**: `sentence-transformers/all-mpnet-base-v2` (768維)
    *   HuggingFace 排行榜前列的通用模型，對語義理解遠強於 MiniLM。
*   **Vector Store**: `ChromaDB`
    *   使用 Cosine Similarity 進行初步篩選。
*   **Chunking**: 
    *   **Context Injection**: 自動將圖片的語義摘要 (Context) 注入到數據表格 (Data) 中，確保數據塊具備可檢索性。
    *   保留 Vision 生成的 Q&A 作為高優先級片段。

### 3. 重排序 (Reranking)
*   **模型**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
*   **機制**: 對初步檢索的 Top-100 結果進行深度比對，重新排序並選出 Top-15 最相關片段給 LLM。

### 4. 生成 (Generation)
*   **LLM**: `ollama/mistral:7b` (or `llama3.2:3b` for speed)
*   **Prompting**: Chain of Thought (CoT) + Few-Shot Learning (動態示例注入)。

---

## 🛠️ 安裝與設置 (Setup)

### 1. 環境準備
```bash
# 建立虛擬環境
python -m venv venv
.\venv\Scripts\activate

# 安裝依賴
pip install -r requirements.txt
```

### 2. 安裝 Ollama 模型
請確保已安裝 [Ollama](https://ollama.com/) 並執行以下指令拉取模型：
```bash
ollama pull llama3.2-vision  # 用於圖片處理
ollama pull mistral          # 用於回答問題 (或 llama3.2)
```

---

## ▶️ 執行流程 (Pipeline Execution)

整個 RAG 流程分為三個主要步驟，必須依序執行：

### Step 1: PDF 解析與轉換
使用 `marker-pdf` 將原始 PDF 轉換為 Markdown 格式，並提取圖片。
```bash
python scripts/pipeline_step1_marker.py
```

### Step 2: 視覺數據提取 (Vision Extraction)
**這是最關鍵的一步。** 使用 Vision 模型掃描所有圖片，生成語義摘要 (Context) 和 問答對 (Q&A)。
*注意：這一步需要較長時間。*
```bash
python scripts/pipeline_step2_vision.py
```

### Step 3: 建立索引 (Indexing)
將處理好的文本和圖片描述切分 (Chunking)，生成 Embedding，並存入 ChromaDB。
*   包含 Context Injection 和 Q&A 優先級處理。
*   首次執行會自動下載 MPNet 模型。
```bash
python scripts/pipeline_step3_index.py
```

### Step 4: 訓練與評估 (Evaluation)
使用訓練集 (train_QA.csv) 評估系統準確率。
```bash
python scripts/run_train.py
```

---

## 📂 專案結構 (Structure)

```
my_rag_project/
├── artifacts/           # 產生的中間檔案
│   ├── processed_docs/  # Markdown 和圖片
│   └── chroma_db/       # 向量資料庫
├── data/                # 原始數據 (PDFs, CSVs)
├── scripts/             # 程式碼
│   ├── core/            # 核心模組 (Config, RAG, VectorStore)
│   │   ├── config.py
│   │   ├── rag_pipeline.py
│   │   └── vector_store.py
│   ├── pipeline_step1_marker.py
│   ├── pipeline_step2_vision.py
│   └── pipeline_step3_index.py
└── requirements.txt
```

## 🔧 常見問題 (Troubleshooting)

*   **OOM (Out of Memory)**: 如果遇到顯存不足，請在 `scripts/core/config.py` 中切換較小的 LLM (如 `llama3.2:3b`) 或減少 `BATCH_SIZE`。
*   **檢索不準**: 確保已完整執行 `pipeline_step2_vision.py`，因為這是數據源頭。
*   **模型下載失敗**: 請檢查網路連線，HuggingFace 模型初次下載需要時間。
