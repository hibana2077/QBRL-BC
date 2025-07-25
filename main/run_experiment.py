"""
快速運行腳本
簡化版的實驗執行程序
"""

import sys
import os

# 添加當前目錄到Python路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiment_pipeline import ExperimentPipeline


def main():
    """主函數"""
    print("QBRL-BC: 量化分箱與非負CP張量分解乳癌診斷模型")
    print("=" * 60)
    
    # 檢查資料文件是否存在
    data_path = "dataset/UCI_BCD.csv"
    if not os.path.exists(data_path):
        print(f"錯誤：找不到資料文件 {data_path}")
        print("請確保UCI_BCD.csv位於dataset/目錄中")
        return
    
    try:
        # 創建並運行實驗管道
        pipeline = ExperimentPipeline(
            data_path=data_path,
            output_dir="results",
            random_state=42
        )
        
        # 運行實驗
        results = pipeline.run_full_experiment()
        
        if results:
            print("\n" + "=" * 60)
            print("實驗成功完成！")
            print("=" * 60)
            print("結果文件：")
            print("  - results/experiment_results.json")
            print("  - results/experiment_report.txt") 
            print("  - results/decision_curves.png")
            print("\n請查看results目錄獲取詳細結果。")
        else:
            print("\n實驗執行失敗！")
            
    except Exception as e:
        print(f"\n運行過程中發生錯誤：{e}")
        print("請檢查數據文件和依賴是否正確安裝。")


if __name__ == "__main__":
    main()
