# QBRL-BC: 量化分箱與非負CP張量分解的乳癌診斷模型

基於量化分箱和非負CP張量分解的可解釋乳癌診斷模型，適用於CPU環境。

## 專案概述

本專案實現了一個創新的乳癌診斷模型，主要特點：
- 🔬 **量化分箱**：將連續特徵轉換為可解釋的區間
- 🧮 **張量分解**：使用非負CP分解提取潛在模式
- 📋 **規則提取**：生成可解釋的診斷規則
- 💻 **CPU友善**：專為CPU環境優化，無需GPU
- 📊 **全面評估**：包含決策曲線分析(DCA)

## 文件結構

```
QBRL-BC/
├── main/
│   ├── dataset/
│   │   └── UCI_BCD.csv              # UCI乳癌數據集
│   ├── data_preprocessing.py        # 數據預處理模組
│   ├── quantization_binning.py      # 量化分箱模組
│   ├── tensor_builder.py           # 張量構建模組
│   ├── cp_decomposition.py         # 非負CP分解模組
│   ├── rule_extraction.py          # 規則提取模組
│   ├── model_evaluation.py         # 模型評估模組
│   ├── decision_curve_analysis.py  # 決策曲線分析模組
│   ├── experiment_pipeline.py      # 實驗管道主程式
│   └── run_experiment.py           # 快速運行腳本
├── docs/
│   └── quest.md                     # 研究計畫文檔
├── requirements.txt                 # Python依賴
└── README.md                       # 本文件
```

## 快速開始

### 1. 環境設置

```bash
# 創建虛擬環境（可選）
python -m venv qbrl_env
source qbrl_env/bin/activate  # Windows: qbrl_env\Scripts\activate

# 安裝依賴
pip install -r requirements.txt
```

### 2. 運行實驗

```bash
cd main
python run_experiment.py
```

### 3. 查看結果

實驗完成後，結果保存在 `results/` 目錄：
- `experiment_results.json`: 詳細實驗數據
- `experiment_report.txt`: 文字報告
- `decision_curves.png`: 決策曲線圖

## 模組說明

### 1. 數據預處理 (`data_preprocessing.py`)
- 載入UCI乳癌數據集
- 數據品質檢查
- 特徵標準化
- 交叉驗證分割

### 2. 量化分箱 (`quantization_binning.py`)
- **等寬分箱**：將特徵值域等分
- **等頻分箱**：每個區間包含相同數量樣本
- **MDLP監督式分箱**：基於資訊增益的最佳分割

### 3. 張量構建 (`tensor_builder.py`)
- 構建三維張量（樣本×特徵×分箱）
- 支援One-Hot、計數、加權編碼
- 張量壓縮與統計分析

### 4. 非負CP分解 (`cp_decomposition.py`)
- 實現CPU友善的非負CP分解算法
- 交替最小二乘法（ALS）優化
- 因子正規化與排序

### 5. 規則提取 (`rule_extraction.py`)
- 將CP因子轉換為IF-THEN規則
- 規則品質評估與過濾
- 生成可解釋的診斷特徵

### 6. 模型評估 (`model_evaluation.py`)
- 多種分類器訓練（邏輯迴歸、隨機森林等）
- 全面性能評估（AUC、F1、Brier Score等）
- McNemar檢定比較模型

### 7. 決策曲線分析 (`decision_curve_analysis.py`)
- 計算淨效益曲線
- 臨床決策價值評估
- 最佳決策閾值選擇

## 實驗流程

1. **參數搜索**：系統性測試不同分箱策略、張量方法和CP秩
2. **最佳配置選擇**：基於AUC選擇最佳參數組合
3. **完整評估**：使用最佳配置訓練所有模型
4. **決策曲線分析**：評估臨床應用價值
5. **結果保存**：生成詳細報告和可視化圖表

## 技術特點

- ✅ **可解釋性**：規則明確，易於臨床理解
- ✅ **CPU效率**：無需GPU，適合資源受限環境
- ✅ **模組化設計**：每個文件專注單一功能，易於維護
- ✅ **全面評估**：包含分類性能、校準度、臨床價值
- ✅ **穩健性**：多種分箱策略和基準模型比較

## 研究貢獻

1. **張量分解導向規則化解釋**：創新性地將分箱資料轉換為張量並提取可讀模式
2. **低資源可重現性**：全流程基於CPU實現，適合實際應用
3. **分箱策略×張量分解交互研究**：系統性比較不同組合的效果
4. **臨床價值評估**：結合DCA評估實際臨床決策效益

## 引用

如果您使用本專案，請引用：

```
QBRL-BC: A Quantization-Based Binning and Non-Negative CP Tensor Decomposition Model 
for Interpretable and CPU-Efficient Breast Cancer Diagnosis
```

## 聯絡方式

如有問題或建議，請通過Issues或Pull Requests聯絡。