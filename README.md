# 互動式線性迴歸視覺化工具

這是一個使用 Streamlit 建立的互動式網頁應用程式，旨在透過視覺化的方式幫助使用者理解簡易線性迴歸。

## 功能

- **互動式參數調整:** 使用者可以透過側邊欄的滑桿動態調整以下參數：
  - **斜率 (a):** 控制線性關係的斜率。
  - **雜訊 (noise):** 控制資料點的分散程度。
  - **資料點數量 (n):** 控制樣本數的大小。
- **即時視覺化:** 參數調整後，圖表會即時更新，顯示新的資料點分佈和對應的迴歸線。
- **CRISP-DM 框架:** 整個應用程式的流程遵循 CRISP-DM (跨產業資料探勘標準流程) 的步驟，包含商業理解、資料理解、資料準備、模型建立、模型評估和部署。

## 如何執行

1. **安裝依賴套件:**
   ```bash
   pip install -r requirements.txt
   ```

2. **執行 Streamlit 應用程式:**
   ```bash
   streamlit run app.py
   ```

Windows (PowerShell) 示例：

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
streamlit run app.py
```

參考老師的範例與部署：

- 原始碼範例: https://github.com/huanchen1107/20250920_AutoDeployLR
- 線上示範: https://aiotda.streamlit.app/
