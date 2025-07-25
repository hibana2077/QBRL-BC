"""
量化分箱模組
實現等寬、等頻和監督式MDLP分箱策略
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')


class QuantizationBinner:
    """量化分箱器"""
    
    def __init__(self, strategy='equal_width', n_bins=5):
        """
        初始化分箱器
        
        Args:
            strategy: 分箱策略 ('equal_width', 'equal_frequency', 'mdlp')
            n_bins: 分箱數量（對MDLP無效）
        """
        self.strategy = strategy
        self.n_bins = n_bins
        self.bin_edges_ = {}
        self.feature_names = None
        
    def fit_transform(self, X, y=None, feature_names=None):
        """擬合並轉換數據"""
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        if self.strategy == 'equal_width':
            return self._equal_width_binning(X)
        elif self.strategy == 'equal_frequency':
            return self._equal_frequency_binning(X)
        elif self.strategy == 'mdlp':
            if y is None:
                raise ValueError("MDLP分箱需要提供標籤y")
            return self._mdlp_binning(X, y)
        else:
            raise ValueError(f"未知的分箱策略：{self.strategy}")
    
    def transform(self, X):
        """使用已學習的分箱邊界轉換新數據"""
        X_binned = np.zeros_like(X, dtype=int)
        
        for i in range(X.shape[1]):
            edges = self.bin_edges_[i]
            X_binned[:, i] = np.digitize(X[:, i], edges) - 1
            # 確保分箱索引在有效範圍內
            X_binned[:, i] = np.clip(X_binned[:, i], 0, len(edges) - 2)
        
        return X_binned
    
    def _equal_width_binning(self, X):
        """等寬分箱"""
        X_binned = np.zeros_like(X, dtype=int)
        
        for i in range(X.shape[1]):
            min_val, max_val = X[:, i].min(), X[:, i].max()
            # 避免除零
            if max_val == min_val:
                edges = [min_val - 0.1, max_val + 0.1]
            else:
                edges = np.linspace(min_val, max_val, self.n_bins + 1)
            
            self.bin_edges_[i] = edges
            X_binned[:, i] = np.digitize(X[:, i], edges) - 1
            X_binned[:, i] = np.clip(X_binned[:, i], 0, self.n_bins - 1)
        
        print(f"等寬分箱完成，每個特徵 {self.n_bins} 個分箱")
        return X_binned
    
    def _equal_frequency_binning(self, X):
        """等頻分箱"""
        X_binned = np.zeros_like(X, dtype=int)
        
        for i in range(X.shape[1]):
            # 計算分位數
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            edges = np.quantile(X[:, i], quantiles)
            # 移除重複邊界
            edges = np.unique(edges)
            
            self.bin_edges_[i] = edges
            X_binned[:, i] = np.digitize(X[:, i], edges) - 1
            X_binned[:, i] = np.clip(X_binned[:, i], 0, len(edges) - 2)
        
        print(f"等頻分箱完成，目標每個特徵 {self.n_bins} 個分箱")
        return X_binned
    
    def _mdlp_binning(self, X, y):
        """MDLP監督式分箱"""
        X_binned = np.zeros_like(X, dtype=int)
        
        for i in range(X.shape[1]):
            edges = self._mdlp_discretize(X[:, i], y)
            self.bin_edges_[i] = edges
            
            X_binned[:, i] = np.digitize(X[:, i], edges) - 1
            X_binned[:, i] = np.clip(X_binned[:, i], 0, len(edges) - 2)
        
        avg_bins = np.mean([len(edges) - 1 for edges in self.bin_edges_.values()])
        print(f"MDLP分箱完成，平均每個特徵 {avg_bins:.1f} 個分箱")
        return X_binned
    
    def _mdlp_discretize(self, feature, target, min_samples_leaf=5):
        """使用決策樹實現簡化的MDLP分箱"""
        # 使用決策樹找到最佳分割點
        dt = DecisionTreeClassifier(
            criterion='entropy',
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        feature_reshaped = feature.reshape(-1, 1)
        dt.fit(feature_reshaped, target)
        
        # 提取分割閾值
        tree = dt.tree_
        thresholds = []
        
        def extract_thresholds(node_id):
            if tree.children_left[node_id] != tree.children_right[node_id]:
                threshold = tree.threshold[node_id]
                thresholds.append(threshold)
                extract_thresholds(tree.children_left[node_id])
                extract_thresholds(tree.children_right[node_id])
        
        extract_thresholds(0)
        
        # 構建分箱邊界
        if not thresholds:
            edges = [feature.min() - 0.1, feature.max() + 0.1]
        else:
            thresholds = sorted(thresholds)
            edges = [feature.min() - 0.1] + thresholds + [feature.max() + 0.1]
        
        return np.array(edges)
    
    def get_bin_info(self):
        """獲取分箱信息"""
        info = {}
        for i, feature_name in enumerate(self.feature_names):
            edges = self.bin_edges_[i]
            n_bins = len(edges) - 1
            info[feature_name] = {
                'n_bins': n_bins,
                'edges': edges.tolist(),
                'ranges': [(edges[j], edges[j+1]) for j in range(n_bins)]
            }
        return info


if __name__ == "__main__":
    # 測試分箱器
    from data_preprocessing import load_and_preprocess_data
    
    X, y, feature_names = load_and_preprocess_data("dataset/UCI_BCD.csv")
    
    if X is not None:
        # 測試不同分箱策略
        strategies = ['equal_width', 'equal_frequency', 'mdlp']
        
        for strategy in strategies:
            print(f"\n測試 {strategy} 分箱策略：")
            binner = QuantizationBinner(strategy=strategy, n_bins=5)
            X_binned = binner.fit_transform(X, y, feature_names)
            print(f"分箱後數據形狀：{X_binned.shape}")
            print(f"分箱值範圍：{X_binned.min()} - {X_binned.max()}")
