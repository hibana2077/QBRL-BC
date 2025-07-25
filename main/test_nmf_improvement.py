"""
NMF + 互動項設計測試腳本
測試新實現的NMF方法是否能有效改進模型性能
"""

import sys
import os
import numpy as np
import pandas as pd

# 添加當前目錄到Python路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import DataPreprocessor
from quantization_binning import QuantizationBinner  
from nmf_interaction_design import NMFInteractionDesign
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def test_nmf_interaction_design():
    """測試NMF + 互動項設計"""
    print("=" * 60)
    print("NMF + 互動項設計測試")
    print("=" * 60)
    
    # 1. 載入資料
    print("\n1. 載入資料...")
    data_path = "dataset/UCI_BCD.csv"
    
    if not os.path.exists(data_path):
        print(f"錯誤：找不到資料文件 {data_path}")
        return False
    
    preprocessor = DataPreprocessor(data_path)
    X, y = preprocessor.load_data()
    
    if X is None or y is None:
        print("資料載入失敗")
        return False
    
    # 檢查並標準化特徵
    if preprocessor.check_data_quality(X, y):
        X_scaled = preprocessor.standardize_features(X)
        print(f"資料載入成功：{X.shape[0]} 樣本，{X.shape[1]} 特徵")
        print(f"正樣本：{np.sum(y)}，負樣本：{np.sum(1-y)}")
    else:
        print("資料品質檢查失敗")
        return False
    
    # 2. 分割資料
    print("\n2. 分割訓練和測試資料...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"訓練集：{X_train.shape[0]} 樣本")
    print(f"測試集：{X_test.shape[0]} 樣本")
    
    # 3. 量化分箱
    print("\n3. 執行量化分箱...")
    
    # 測試不同分箱策略
    binning_strategies = ['equal_width', 'equal_frequency', 'mdlp']
    bin_counts = [3, 5, 7]
    
    best_score = 0
    best_config = None
    best_model = None
    
    for strategy in binning_strategies:
        for n_bins in bin_counts:
            print(f"\n  測試配置：{strategy}, {n_bins} bins")
            
            try:
                # 分箱
                binner = QuantizationBinner(strategy=strategy, n_bins=n_bins)
                X_train_binned = binner.fit_transform(X_train, y_train, preprocessor.feature_names)
                X_test_binned = binner.transform(X_test)
                
                print(f"    分箱完成，形狀：{X_train_binned.shape}")
                
                # 4. 測試不同NMF配置
                nmf_configs = [
                    {'n_components': 5, 'alpha': 0.1},
                    {'n_components': 10, 'alpha': 0.1},
                    {'n_components': 15, 'alpha': 0.01},
                ]
                
                for config in nmf_configs:
                    print(f"    NMF配置：{config}")
                    
                    # 創建並訓練NMF模型
                    nmf_model = NMFInteractionDesign(
                        n_components=config['n_components'],
                        max_iter=200,
                        random_state=42,
                        alpha=config['alpha']
                    )
                    
                    # 訓練模型
                    nmf_model.fit(
                        X_train_binned, y_train,
                        feature_names=preprocessor.feature_names,
                        bin_info=binner.get_bin_info(),
                        add_interactions=True,
                        classifier_type='logistic'
                    )
                    
                    # 預測
                    y_pred_proba = nmf_model.predict_proba(X_test_binned)[:, 1]
                    y_pred = nmf_model.predict(X_test_binned)
                    
                    # 評估
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                    print(f"    AUC分數：{auc_score:.4f}")
                    
                    # 更新最佳配置
                    if auc_score > best_score:
                        best_score = auc_score
                        best_config = {
                            'binning_strategy': strategy,
                            'n_bins': n_bins,
                            'nmf_components': config['n_components'],
                            'nmf_alpha': config['alpha']
                        }
                        best_model = nmf_model
                        print(f"    ★ 新的最佳配置！AUC: {best_score:.4f}")
            
            except Exception as e:
                print(f"    配置失敗：{e}")
                continue
    
    # 5. 顯示最佳結果
    print("\n" + "=" * 60)
    print("測試完成 - 最佳結果")
    print("=" * 60)
    
    if best_model is not None:
        print(f"最佳AUC分數：{best_score:.4f}")
        print(f"最佳配置：{best_config}")
        
        # 重新評估最佳模型
        binner_final = QuantizationBinner(
            strategy=best_config['binning_strategy'], 
            n_bins=best_config['n_bins']
        )
        X_train_binned_final = binner_final.fit_transform(X_train, y_train, preprocessor.feature_names)
        X_test_binned_final = binner_final.transform(X_test)
        
        y_pred_final = best_model.predict(X_test_binned_final)
        y_pred_proba_final = best_model.predict_proba(X_test_binned_final)[:, 1]
        
        print("\n詳細分類報告：")
        print(classification_report(y_test, y_pred_final))
        
        print(f"\n最終評估指標：")
        print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba_final):.4f}")
        
        # 顯示模型摘要
        model_summary = best_model.get_model_summary()
        print(f"\n模型摘要：")
        for key, value in model_summary.items():
            if key != 'component_interpretations':
                print(f"  {key}: {value}")
        
        # 顯示前3個分量的解釋
        print(f"\n前3個NMF分量解釋：")
        for i, comp in enumerate(model_summary.get('component_interpretations', [])[:3]):
            print(f"  分量 {i+1} (稀疏度: {comp['sparsity']:.3f}):")
            for feature_name, weight in comp['top_features'][:5]:
                print(f"    {feature_name}: {weight:.3f}")
        
        return True
    else:
        print("所有配置都失敗了")
        return False


def compare_with_baseline():
    """與基準模型比較"""
    print("\n" + "=" * 60)
    print("與基準模型比較")
    print("=" * 60)
    
    # 載入資料
    data_path = "dataset/UCI_BCD.csv"
    preprocessor = DataPreprocessor(data_path)
    X, y = preprocessor.load_data()
    X_scaled = preprocessor.standardize_features(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 基準模型
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    
    baseline_models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    print("基準模型結果：")
    baseline_scores = {}
    for name, model in baseline_models.items():
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        baseline_scores[name] = auc
        print(f"  {name}: AUC = {auc:.4f}")
    
    # NMF模型
    binner = QuantizationBinner(strategy='mdlp', n_bins=5)
    X_train_binned = binner.fit_transform(X_train, y_train, preprocessor.feature_names)
    X_test_binned = binner.transform(X_test)
    
    nmf_model = NMFInteractionDesign(
        n_components=10,
        max_iter=200,
        random_state=42,
        alpha=0.1
    )
    
    nmf_model.fit(
        X_train_binned, y_train,
        feature_names=preprocessor.feature_names,
        bin_info=binner.get_bin_info(),
        add_interactions=True,
        classifier_type='logistic'
    )
    
    y_pred_proba_nmf = nmf_model.predict_proba(X_test_binned)[:, 1]
    auc_nmf = roc_auc_score(y_test, y_pred_proba_nmf)
    
    print(f"\nNMF + 互動項設計: AUC = {auc_nmf:.4f}")
    
    # 比較結果
    print(f"\n性能改進：")
    for name, baseline_auc in baseline_scores.items():
        improvement = ((auc_nmf - baseline_auc) / baseline_auc) * 100
        print(f"  vs {name}: {improvement:+.2f}%")


if __name__ == "__main__":
    try:
        # 測試NMF方法
        success = test_nmf_interaction_design()
        
        if success:
            # 與基準模型比較
            compare_with_baseline()
        
        print("\n" + "=" * 60)
        print("測試完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n測試過程中發生錯誤：{e}")
        import traceback
        traceback.print_exc()
