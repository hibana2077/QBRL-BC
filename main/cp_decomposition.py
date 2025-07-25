"""
非負CP張量分解模組
實現CPU友善的非負CP分解演算法
"""

import numpy as np
from scipy.optimize import nnls
import warnings
warnings.filterwarnings('ignore')


class NonNegativeCPDecomposition:
    """非負CP張量分解器"""
    
    def __init__(self, rank=5, max_iter=100, tol=1e-6, random_state=42):
        """
        初始化CP分解器
        
        Args:
            rank: 分解秩（因子數量）
            max_iter: 最大迭代次數
            tol: 收斂容忍度
            random_state: 隨機種子
        """
        self.rank = rank
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # 分解結果
        self.factors_ = None
        self.weights_ = None
        self.reconstruction_error_ = []
        self.converged_ = False
        
    def fit(self, tensor):
        """
        擬合CP分解
        
        Args:
            tensor: 三維張量 (I, J, K)
        
        Returns:
            self
        """
        np.random.seed(self.random_state)
        
        I, J, K = tensor.shape
        
        # 初始化因子矩陣
        A = np.random.rand(I, self.rank)
        B = np.random.rand(J, self.rank)
        C = np.random.rand(K, self.rank)
        
        # 交替最小二乘法（ALS）
        prev_error = float('inf')
        
        for iteration in range(self.max_iter):
            # 更新因子A
            A = self._update_factor_A(tensor, A, B, C)
            
            # 更新因子B  
            B = self._update_factor_B(tensor, A, B, C)
            
            # 更新因子C
            C = self._update_factor_C(tensor, A, B, C)
            
            # 計算重建誤差
            error = self._compute_reconstruction_error(tensor, A, B, C)
            self.reconstruction_error_.append(error)
            
            # 檢查收斂
            if abs(prev_error - error) < self.tol:
                self.converged_ = True
                print(f"CP分解在第 {iteration+1} 次迭代收斂")
                break
            
            prev_error = error
        
        if not self.converged_:
            print(f"CP分解在 {self.max_iter} 次迭代後未收斂")
        
        # 正規化因子
        A, B, C, weights = self._normalize_factors(A, B, C)
        
        self.factors_ = [A, B, C]
        self.weights_ = weights
        
        print(f"CP分解完成：秩={self.rank}，最終誤差={self.reconstruction_error_[-1]:.6f}")
        
        return self
    
    def _update_factor_A(self, tensor, A, B, C):
        """更新因子矩陣A"""
        I, J, K = tensor.shape
        A_new = np.zeros_like(A)
        
        # Khatri-Rao乘積
        KR_BC = self._khatri_rao(B, C)
        
        for i in range(I):
            # 展開張量的第i個纖維
            fiber = tensor[i, :, :].flatten()
            
            # 非負最小二乘
            A_new[i, :], _ = nnls(KR_BC, fiber)
        
        return A_new
    
    def _update_factor_B(self, tensor, A, B, C):
        """更新因子矩陣B"""
        I, J, K = tensor.shape
        B_new = np.zeros_like(B)
        
        # Khatri-Rao乘積
        KR_AC = self._khatri_rao(A, C)
        
        for j in range(J):
            # 展開張量的第j個纖維
            fiber = tensor[:, j, :].flatten()
            
            # 非負最小二乘
            B_new[j, :], _ = nnls(KR_AC, fiber)
        
        return B_new
    
    def _update_factor_C(self, tensor, A, B, C):
        """更新因子矩陣C"""
        I, J, K = tensor.shape
        C_new = np.zeros_like(C)
        
        # Khatri-Rao乘積
        KR_AB = self._khatri_rao(A, B)
        
        for k in range(K):
            # 展開張量的第k個纖維
            fiber = tensor[:, :, k].flatten()
            
            # 非負最小二乘
            C_new[k, :], _ = nnls(KR_AB, fiber)
        
        return C_new
    
    def _khatri_rao(self, A, B):
        """計算Khatri-Rao乘積"""
        return np.vstack([np.kron(A[:, r], B[:, r]) for r in range(A.shape[1])]).T
    
    def _compute_reconstruction_error(self, tensor, A, B, C):
        """計算重建誤差"""
        reconstructed = self._reconstruct_tensor(A, B, C)
        return np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)
    
    def _reconstruct_tensor(self, A, B, C):
        """從因子重建張量"""
        I, J, K = A.shape[0], B.shape[0], C.shape[0]
        tensor_reconstructed = np.zeros((I, J, K))
        
        for r in range(self.rank):
            tensor_reconstructed += np.outer(A[:, r], np.outer(B[:, r], C[:, r]).flatten()).reshape(I, J, K)
        
        return tensor_reconstructed
    
    def _normalize_factors(self, A, B, C):
        """正規化因子矩陣"""
        weights = np.zeros(self.rank)
        
        for r in range(self.rank):
            # 計算權重
            norm_A = np.linalg.norm(A[:, r])
            norm_B = np.linalg.norm(B[:, r])
            norm_C = np.linalg.norm(C[:, r])
            
            weight = norm_A * norm_B * norm_C
            weights[r] = weight
            
            # 正規化
            if weight > 0:
                A[:, r] = A[:, r] / norm_A
                B[:, r] = B[:, r] / norm_B
                C[:, r] = C[:, r] / norm_C
        
        # 按權重排序
        sorted_indices = np.argsort(weights)[::-1]
        A = A[:, sorted_indices]
        B = B[:, sorted_indices]
        C = C[:, sorted_indices]
        weights = weights[sorted_indices]
        
        return A, B, C, weights
    
    def transform(self, tensor):
        """將張量轉換為因子表示"""
        if self.factors_ is None:
            raise ValueError("模型尚未擬合，請先調用fit方法")
        
        A, B, C = self.factors_
        I, J, K = tensor.shape
        
        # 計算樣本在每個因子上的投影
        factor_scores = np.zeros((I, self.rank))
        
        for i in range(I):
            for r in range(self.rank):
                # 計算樣本i在因子r上的分數
                score = np.sum(tensor[i, :, :] * np.outer(B[:, r], C[:, r]))
                factor_scores[i, r] = score * self.weights_[r]
        
        return factor_scores
    
    def get_factor_interpretation(self, feature_names=None, bin_info=None, threshold=0.1):
        """解釋因子的含義"""
        if self.factors_ is None:
            raise ValueError("模型尚未擬合")
        
        A, B, C = self.factors_
        interpretations = []
        
        for r in range(self.rank):
            interpretation = {
                'factor_id': r,
                'weight': self.weights_[r],
                'sample_pattern': None,
                'feature_pattern': [],
                'bin_pattern': []
            }
            
            # 特徵模式
            feature_weights = B[:, r]
            significant_features = np.where(feature_weights > threshold)[0]
            
            for feat_idx in significant_features:
                feat_name = feature_names[feat_idx] if feature_names else f"Feature_{feat_idx}"
                interpretation['feature_pattern'].append({
                    'feature': feat_name,
                    'weight': feature_weights[feat_idx]
                })
            
            # 分箱模式
            bin_weights = C[:, r]
            significant_bins = np.where(bin_weights > threshold)[0]
            interpretation['bin_pattern'] = [
                {'bin': int(bin_idx), 'weight': bin_weights[bin_idx]}
                for bin_idx in significant_bins
            ]
            
            interpretations.append(interpretation)
        
        return interpretations
    
    def get_factor_features(self):
        """獲取因子特徵矩陣，用於模型訓練"""
        if self.factors_ is None:
            return None
        
        # 返回樣本的因子分數作為特徵
        # 這需要原始張量，但由於我們在這裡沒有，返回 None
        # 實際應該在 transform 方法被調用後使用
        return None


if __name__ == "__main__":
    # 測試CP分解
    from data_preprocessing import load_and_preprocess_data
    from quantization_binning import QuantizationBinner
    from tensor_builder import TensorBuilder
    
    X, y, feature_names = load_and_preprocess_data("dataset/UCI_BCD.csv")
    
    if X is not None:
        # 創建張量
        binner = QuantizationBinner(strategy='equal_width', n_bins=5)
        X_binned = binner.fit_transform(X, y, feature_names)
        
        builder = TensorBuilder(max_bins=6)
        tensor = builder.build_tensor(X_binned, method='one_hot')
        
        # CP分解
        cp = NonNegativeCPDecomposition(rank=3, max_iter=50)
        cp.fit(tensor)
        
        # 獲取因子表示
        factor_scores = cp.transform(tensor)
        print(f"\n因子分數形狀：{factor_scores.shape}")
        
        # 解釋因子
        interpretations = cp.get_factor_interpretation(feature_names)
        for i, interp in enumerate(interpretations):
            print(f"\n因子 {i} (權重: {interp['weight']:.3f}):")
            print(f"重要特徵：{[fp['feature'] for fp in interp['feature_pattern'][:3]]}")
