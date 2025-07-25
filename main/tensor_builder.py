"""
張量構建模組
將分箱後的數據轉換為三維張量（樣本 × 特徵 × 分箱）
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')


class TensorBuilder:
    """三維張量構建器"""
    
    def __init__(self, max_bins=10):
        """
        初始化張量構建器
        
        Args:
            max_bins: 最大分箱數，用於統一張量維度
        """
        self.max_bins = max_bins
        self.n_samples = None
        self.n_features = None
        self.actual_bins = None
        
    def build_tensor(self, X_binned, method='one_hot'):
        """
        構建三維張量
        
        Args:
            X_binned: 分箱後的數據 (n_samples, n_features)
            method: 張量填充方法 ('one_hot', 'count', 'weighted')
        
        Returns:
            tensor: 三維張量 (n_samples, n_features, max_bins)
        """
        self.n_samples, self.n_features = X_binned.shape
        self.actual_bins = [np.max(X_binned[:, i]) + 1 for i in range(self.n_features)]
        
        # 初始化張量
        tensor = np.zeros((self.n_samples, self.n_features, self.max_bins))
        
        if method == 'one_hot':
            tensor = self._build_one_hot_tensor(X_binned)
        elif method == 'count':
            tensor = self._build_count_tensor(X_binned)
        elif method == 'weighted':
            tensor = self._build_weighted_tensor(X_binned)
        else:
            raise ValueError(f"未知的張量構建方法：{method}")
        
        print(f"張量構建完成：{tensor.shape}")
        print(f"張量非零元素比例：{np.mean(tensor > 0):.3f}")
        
        return tensor
    
    def _build_one_hot_tensor(self, X_binned):
        """構建One-Hot編碼張量"""
        tensor = np.zeros((self.n_samples, self.n_features, self.max_bins))
        
        for i in range(self.n_samples):
            for j in range(self.n_features):
                bin_idx = X_binned[i, j]
                if bin_idx < self.max_bins:
                    tensor[i, j, bin_idx] = 1.0
        
        return tensor
    
    def _build_count_tensor(self, X_binned):
        """構建計數張量（適用於多重分箱）"""
        tensor = np.zeros((self.n_samples, self.n_features, self.max_bins))
        
        for i in range(self.n_samples):
            for j in range(self.n_features):
                bin_idx = X_binned[i, j]
                if bin_idx < self.max_bins:
                    tensor[i, j, bin_idx] += 1.0
        
        return tensor
    
    def _build_weighted_tensor(self, X_binned):
        """構建加權張量（基於頻率）"""
        tensor = np.zeros((self.n_samples, self.n_features, self.max_bins))
        
        # 計算每個分箱的權重（基於逆頻率）
        bin_weights = {}
        for j in range(self.n_features):
            unique_bins, counts = np.unique(X_binned[:, j], return_counts=True)
            total_count = len(X_binned[:, j])
            weights = {bin_val: total_count / count for bin_val, count in zip(unique_bins, counts)}
            bin_weights[j] = weights
        
        for i in range(self.n_samples):
            for j in range(self.n_features):
                bin_idx = X_binned[i, j]
                if bin_idx < self.max_bins:
                    weight = bin_weights[j].get(bin_idx, 1.0)
                    tensor[i, j, bin_idx] = weight
        
        # 正規化
        tensor = tensor / np.max(tensor)
        
        return tensor
    
    def get_tensor_statistics(self, tensor):
        """獲取張量統計信息"""
        stats = {
            'shape': tensor.shape,
            'total_elements': tensor.size,
            'non_zero_elements': np.sum(tensor > 0),
            'sparsity': 1.0 - np.mean(tensor > 0),
            'mean_value': np.mean(tensor),
            'std_value': np.std(tensor),
            'max_value': np.max(tensor),
            'min_value': np.min(tensor)
        }
        
        # 每個維度的統計
        stats['sample_activation'] = np.mean(np.sum(tensor, axis=(1, 2)) > 0)
        stats['feature_activation'] = np.mean(np.sum(tensor, axis=(0, 2)) > 0)
        stats['bin_activation'] = np.mean(np.sum(tensor, axis=(0, 1)) > 0)
        
        return stats
    
    def visualize_tensor_slice(self, tensor, sample_idx=0):
        """可視化張量的一個樣本切片"""
        if sample_idx >= tensor.shape[0]:
            print(f"樣本索引 {sample_idx} 超出範圍")
            return None
        
        slice_2d = tensor[sample_idx, :, :]
        print(f"\n樣本 {sample_idx} 的張量切片：")
        print(f"形狀：{slice_2d.shape}")
        print(f"非零元素：{np.sum(slice_2d > 0)}")
        
        # 顯示每個特徵的激活分箱
        for i in range(min(5, slice_2d.shape[0])):  # 只顯示前5個特徵
            active_bins = np.where(slice_2d[i, :] > 0)[0]
            if len(active_bins) > 0:
                print(f"特徵 {i}: 激活分箱 {active_bins.tolist()}")
        
        return slice_2d
    
    def compress_tensor(self, tensor, method='remove_empty_bins'):
        """壓縮張量以減少記憶體使用"""
        if method == 'remove_empty_bins':
            # 移除完全為空的分箱
            active_bins = np.any(tensor > 0, axis=(0, 1))
            if np.sum(active_bins) < tensor.shape[2]:
                tensor_compressed = tensor[:, :, active_bins]
                print(f"移除空分箱：{tensor.shape[2]} -> {tensor_compressed.shape[2]}")
                return tensor_compressed, active_bins
        
        return tensor, None


if __name__ == "__main__":
    # 測試張量構建
    from data_preprocessing import load_and_preprocess_data
    from quantization_binning import QuantizationBinner
    
    X, y, feature_names = load_and_preprocess_data("dataset/UCI_BCD.csv")
    
    if X is not None:
        # 分箱
        binner = QuantizationBinner(strategy='equal_width', n_bins=5)
        X_binned = binner.fit_transform(X, y, feature_names)
        
        # 構建張量
        builder = TensorBuilder(max_bins=8)
        tensor = builder.build_tensor(X_binned, method='one_hot')
        
        # 獲取統計信息
        stats = builder.get_tensor_statistics(tensor)
        print(f"\n張量統計：")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # 可視化一個切片
        builder.visualize_tensor_slice(tensor, sample_idx=0)
