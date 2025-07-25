"""
數據預處理模組
負責加載和清理乳癌診斷資料集
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """數據預處理類別"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self):
        """加載UCI乳癌資料集"""
        try:
            # 讀取CSV文件，處理可能的格式問題
            df = pd.read_csv(self.data_path)
            
            # 如果第一行有格式問題，重新讀取
            if df.shape[1] > 32:  # 正常應該有32列
                # 跳過可能有問題的行，重新讀取
                df = pd.read_csv(self.data_path, skiprows=1, 
                    names=['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                           'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean',
                           'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se',
                           'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se',
                           'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
                           'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
                           'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'])
            
            # 移除ID列
            if 'id' in df.columns:
                df = df.drop('id', axis=1)
            
            # 分離特徵和標籤
            X = df.drop('diagnosis', axis=1)
            y = df['diagnosis']
            
            # 將標籤轉換為數值（M=1, B=0）
            y = (y == 'M').astype(int)
            
            self.feature_names = X.columns.tolist()
            
            print(f"資料載入成功：{X.shape[0]} 個樣本，{X.shape[1]} 個特徵")
            print(f"惡性樣本數：{y.sum()}，良性樣本數：{(1-y).sum()}")
            
            return X.values, y.values
            
        except Exception as e:
            print(f"資料載入失敗：{e}")
            return None, None
    
    def check_data_quality(self, X, y):
        """檢查數據品質"""
        # 檢查缺失值
        missing_count = np.sum(np.isnan(X))
        print(f"缺失值數量：{missing_count}")
        
        # 檢查無限值
        inf_count = np.sum(np.isinf(X))
        print(f"無限值數量：{inf_count}")
        
        # 檢查重複樣本
        unique_samples = len(np.unique(X, axis=0))
        print(f"唯一樣本數：{unique_samples} / {X.shape[0]}")
        
        return missing_count == 0 and inf_count == 0
    
    def standardize_features(self, X_train, X_test=None):
        """標準化特徵"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def create_cv_splits(self, X, y, n_splits=5, random_state=42):
        """創建分層交叉驗證分割"""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        splits = []
        for train_idx, test_idx in skf.split(X, y):
            splits.append((train_idx, test_idx))
        
        print(f"創建了 {n_splits} 折交叉驗證分割")
        return splits


def load_and_preprocess_data(data_path):
    """便捷函數：載入和預處理數據"""
    preprocessor = DataPreprocessor(data_path)
    X, y = preprocessor.load_data()
    
    if X is not None and y is not None:
        # 檢查數據品質
        is_clean = preprocessor.check_data_quality(X, y)
        
        if is_clean:
            print("數據品質檢查通過")
            return X, y, preprocessor.feature_names
        else:
            print("數據品質檢查失敗")
            return None, None, None
    
    return None, None, None


if __name__ == "__main__":
    # 測試數據預處理
    data_path = "dataset/UCI_BCD.csv"
    X, y, feature_names = load_and_preprocess_data(data_path)
    
    if X is not None:
        print(f"\n特徵名稱（前10個）：{feature_names[:10]}")
        print(f"數據形狀：X={X.shape}, y={y.shape}")
