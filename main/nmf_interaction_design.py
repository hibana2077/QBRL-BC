"""
NMF + 互動項設計模組
實現將「樣本 × (特徵×分箱)」矩陣做 NMF，得到部件式解釋，再接分類器
作為CP張量分解的備用技術方案
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class NMFInteractionDesign:
    """NMF + 互動項設計類別"""
    
    def __init__(self, n_components=10, max_iter=200, random_state=42, alpha=0.1):
        """
        初始化NMF互動項設計器
        
        Args:
            n_components: NMF分量數量
            max_iter: 最大迭代次數
            random_state: 隨機種子
            alpha: L1正則化係數
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        
        # 模型組件
        self.nmf_model = None
        self.scaler = StandardScaler()
        self.classifier = None
        
        # 訓練後的屬性
        self.feature_matrix = None
        self.interaction_matrix = None
        self.nmf_features = None
        self.feature_names = None
        self.bin_info = None
        
        # 可解釋性資訊
        self.component_interpretations = []
        self.feature_importance = None
        
    def _create_interaction_matrix(self, X_binned, feature_names=None, bin_info=None):
        """
        創建特徵與分箱的互動項矩陣
        
        Args:
            X_binned: 分箱後的數據 (n_samples, n_features)
            feature_names: 特徵名稱列表
            bin_info: 分箱資訊字典
            
        Returns:
            interaction_matrix: 互動項矩陣 (n_samples, n_features * n_bins)
        """
        n_samples, n_features = X_binned.shape
        
        # 儲存原始資訊
        self.feature_names = feature_names if feature_names else [f"feature_{i}" for i in range(n_features)]
        self.bin_info = bin_info
        
        # 計算每個特徵的最大分箱數
        max_bins_per_feature = []
        for i in range(n_features):
            max_bin = int(np.max(X_binned[:, i])) + 1
            max_bins_per_feature.append(max_bin)
        
        # 創建互動項特徵名稱
        interaction_feature_names = []
        for i, feature_name in enumerate(self.feature_names):
            for bin_idx in range(max_bins_per_feature[i]):
                interaction_feature_names.append(f"{feature_name}_bin_{bin_idx}")
        
        # 創建互動項矩陣
        total_interactions = sum(max_bins_per_feature)
        interaction_matrix = np.zeros((n_samples, total_interactions))
        
        col_idx = 0
        for feature_idx in range(n_features):
            for bin_idx in range(max_bins_per_feature[feature_idx]):
                # 對於每個樣本，如果該特徵落在該分箱，則設為1
                mask = (X_binned[:, feature_idx] == bin_idx)
                interaction_matrix[mask, col_idx] = 1.0
                col_idx += 1
        
        # 儲存互動項特徵名稱
        self.interaction_feature_names = interaction_feature_names
        
        return interaction_matrix
    
    def _add_feature_interactions(self, interaction_matrix):
        """
        添加特徵間的二階互動項
        
        Args:
            interaction_matrix: 基礎互動項矩陣
            
        Returns:
            enhanced_matrix: 增強的互動項矩陣
        """
        n_samples, n_base_features = interaction_matrix.shape
        
        # 選擇前k個最重要的基礎特徵進行二階互動
        # 這裡簡化為選擇變異數最大的特徵
        feature_variance = np.var(interaction_matrix, axis=0)
        top_k = min(20, n_base_features)  # 限制互動項數量
        top_indices = np.argsort(feature_variance)[-top_k:]
        
        # 創建二階互動項
        second_order_interactions = []
        second_order_names = []
        
        for i in range(len(top_indices)):
            for j in range(i+1, len(top_indices)):
                idx1, idx2 = top_indices[i], top_indices[j]
                interaction = interaction_matrix[:, idx1] * interaction_matrix[:, idx2]
                second_order_interactions.append(interaction)
                
                name1 = self.interaction_feature_names[idx1]
                name2 = self.interaction_feature_names[idx2]
                second_order_names.append(f"{name1}*{name2}")
        
        if second_order_interactions:
            second_order_matrix = np.column_stack(second_order_interactions)
            enhanced_matrix = np.hstack([interaction_matrix, second_order_matrix])
            self.interaction_feature_names.extend(second_order_names)
        else:
            enhanced_matrix = interaction_matrix
        
        return enhanced_matrix
    
    def fit(self, X_binned, y, feature_names=None, bin_info=None, 
            add_interactions=True, classifier_type='logistic'):
        """
        訓練NMF + 互動項模型
        
        Args:
            X_binned: 分箱後的數據
            y: 標籤
            feature_names: 特徵名稱
            bin_info: 分箱資訊
            add_interactions: 是否添加特徵間互動項
            classifier_type: 分類器類型 ('logistic', 'rf')
        """
        print("創建特徵-分箱互動項矩陣...")
        
        # 1. 創建基礎互動項矩陣
        self.interaction_matrix = self._create_interaction_matrix(
            X_binned, feature_names, bin_info
        )
        
        # 2. 添加特徵間互動項（可選）
        if add_interactions:
            print("添加特徵間二階互動項...")
            self.interaction_matrix = self._add_feature_interactions(self.interaction_matrix)
        
        print(f"互動項矩陣形狀: {self.interaction_matrix.shape}")
        
        # 3. 標準化互動項矩陣
        self.interaction_matrix_scaled = self.scaler.fit_transform(self.interaction_matrix)
        
        # NMF要求非負數據，確保數據非負
        # 方法1：使用min-max scaling使數據在[0,1]範圍
        min_val = np.min(self.interaction_matrix_scaled)
        if min_val < 0:
            self.interaction_matrix_scaled = self.interaction_matrix_scaled - min_val
        
        # 確保數據非負且避免全零行
        self.interaction_matrix_scaled = np.maximum(self.interaction_matrix_scaled, 1e-8)
        
        # 4. 應用NMF分解
        print(f"執行NMF分解 (n_components={self.n_components})...")
        self.nmf_model = NMF(
            n_components=self.n_components,
            max_iter=self.max_iter,
            random_state=self.random_state,
            alpha_W=self.alpha,  # W矩陣的L1正則化
            alpha_H=self.alpha,  # H矩陣的L1正則化
            l1_ratio=0.5,  # L1+L2混合正則化
            init='nndsvd'
        )
        
        # 獲取NMF特徵
        self.nmf_features = self.nmf_model.fit_transform(self.interaction_matrix_scaled)
        
        print(f"NMF特徵形狀: {self.nmf_features.shape}")
        print(f"重建誤差: {self.nmf_model.reconstruction_err_:.4f}")
        
        # 5. 解釋NMF分量
        self._interpret_nmf_components()
        
        # 6. 訓練分類器
        print("訓練分類器...")
        if classifier_type == 'logistic':
            self.classifier = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                C=1.0
            )
        elif classifier_type == 'rf':
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=10
            )
        else:
            raise ValueError(f"不支援的分類器類型: {classifier_type}")
        
        self.classifier.fit(self.nmf_features, y)
        
        # 7. 計算特徵重要性
        self._calculate_feature_importance()
        
        return self
    
    def _interpret_nmf_components(self):
        """解釋NMF分量"""
        self.component_interpretations = []
        
        # H矩陣表示每個分量對原始特徵的權重
        H = self.nmf_model.components_
        
        for comp_idx in range(self.n_components):
            # 找出該分量中權重最高的特徵
            component_weights = H[comp_idx, :]
            top_indices = np.argsort(component_weights)[-10:][::-1]  # 前10個
            
            top_features = []
            for idx in top_indices:
                if component_weights[idx] > 0.01:  # 閾值過濾
                    feature_name = self.interaction_feature_names[idx]
                    weight = component_weights[idx]
                    top_features.append((feature_name, weight))
            
            self.component_interpretations.append({
                'component_id': comp_idx,
                'top_features': top_features,
                'sparsity': np.sum(component_weights > 0.01) / len(component_weights)
            })
    
    def _calculate_feature_importance(self):
        """計算特徵重要性"""
        if hasattr(self.classifier, 'feature_importances_'):
            # 隨機森林的特徵重要性
            self.feature_importance = self.classifier.feature_importances_
        elif hasattr(self.classifier, 'coef_'):
            # 邏輯迴歸的係數
            self.feature_importance = np.abs(self.classifier.coef_[0])
        else:
            self.feature_importance = np.ones(self.n_components) / self.n_components
    
    def predict(self, X_binned):
        """預測"""
        if self.nmf_model is None or self.classifier is None:
            raise ValueError("模型尚未訓練")
        
        # 創建互動項矩陣
        interaction_matrix = self._create_interaction_matrix_predict(X_binned)
        
        # 標準化
        interaction_matrix_scaled = self.scaler.transform(interaction_matrix)
        
        # 確保非負（與訓練時相同的處理）
        min_val = np.min(interaction_matrix_scaled)
        if min_val < 0:
            interaction_matrix_scaled = interaction_matrix_scaled - min_val
        interaction_matrix_scaled = np.maximum(interaction_matrix_scaled, 1e-8)
        
        # NMF轉換
        nmf_features = self.nmf_model.transform(interaction_matrix_scaled)
        
        # 分類預測
        return self.classifier.predict(nmf_features)
    
    def predict_proba(self, X_binned):
        """預測機率"""
        if self.nmf_model is None or self.classifier is None:
            raise ValueError("模型尚未訓練")
        
        # 創建互動項矩陣
        interaction_matrix = self._create_interaction_matrix_predict(X_binned)
        
        # 標準化
        interaction_matrix_scaled = self.scaler.transform(interaction_matrix)
        
        # 確保非負（與訓練時相同的處理）
        min_val = np.min(interaction_matrix_scaled)
        if min_val < 0:
            interaction_matrix_scaled = interaction_matrix_scaled - min_val
        interaction_matrix_scaled = np.maximum(interaction_matrix_scaled, 1e-8)
        
        # NMF轉換
        nmf_features = self.nmf_model.transform(interaction_matrix_scaled)
        
        # 分類預測機率
        return self.classifier.predict_proba(nmf_features)
    
    def _create_interaction_matrix_predict(self, X_binned):
        """為預測創建互動項矩陣"""
        n_samples, n_features = X_binned.shape
        
        # 使用訓練時保存的信息來保證維度一致
        if not hasattr(self, 'feature_names') or not hasattr(self, 'interaction_feature_names'):
            raise ValueError("模型尚未訓練或缺少特徵信息")
        
        # 重新創建基礎互動項矩陣，確保與訓練時一致
        interaction_matrix = np.zeros((n_samples, len(self.interaction_feature_names)))
        
        # 解析訓練時的互動項特徵名稱來重建矩陣
        base_feature_count = 0
        
        # 基礎特徵-分箱互動項
        for i, feature_name in enumerate(self.feature_names):
            # 計算該特徵的分箱數（從特徵名稱中推斷）
            feature_bins = [name for name in self.interaction_feature_names if name.startswith(f"{feature_name}_bin_")]
            n_bins_for_feature = len(feature_bins)
            
            for bin_idx in range(n_bins_for_feature):
                if base_feature_count < len(self.interaction_feature_names):
                    feature_interaction_name = f"{feature_name}_bin_{bin_idx}"
                    if feature_interaction_name in self.interaction_feature_names:
                        col_idx = self.interaction_feature_names.index(feature_interaction_name)
                        mask = (X_binned[:, i] == bin_idx)
                        interaction_matrix[mask, col_idx] = 1.0
                        base_feature_count += 1
        
        # 添加二階互動項（如果存在）
        # 找出所有二階互動項
        second_order_names = [name for name in self.interaction_feature_names if '*' in name]
        
        if second_order_names:
            # 重新計算二階互動項
            base_names = [name for name in self.interaction_feature_names if '*' not in name]
            
            for interaction_name in second_order_names:
                if '*' in interaction_name:
                    parts = interaction_name.split('*')
                    if len(parts) == 2:
                        name1, name2 = parts[0], parts[1]
                        if name1 in base_names and name2 in base_names:
                            idx1 = base_names.index(name1)
                            idx2 = base_names.index(name2)
                            col_idx = self.interaction_feature_names.index(interaction_name)
                            
                            # 計算互動項
                            interaction_value = interaction_matrix[:, idx1] * interaction_matrix[:, idx2]
                            interaction_matrix[:, col_idx] = interaction_value
        
        return interaction_matrix
    
    def get_model_summary(self):
        """獲取模型摘要"""
        if self.nmf_model is None:
            return "模型尚未訓練"
        
        summary = {
            'model_type': 'NMF + Interaction Design',
            'n_components': self.n_components,
            'interaction_matrix_shape': self.interaction_matrix.shape,
            'nmf_features_shape': self.nmf_features.shape,
            'reconstruction_error': self.nmf_model.reconstruction_err_,
            'classifier_type': type(self.classifier).__name__,
            'component_interpretations': self.component_interpretations[:3]  # 前3個分量
        }
        
        return summary
    
    def cross_validate(self, X_binned, y, cv=5, scoring='roc_auc'):
        """交叉驗證評估"""
        if self.nmf_model is None:
            raise ValueError("模型尚未訓練")
        
        # 創建完整的特徵流水線
        interaction_matrix = self._create_interaction_matrix_predict(X_binned)
        interaction_matrix_scaled = self.scaler.transform(interaction_matrix)
        
        # 確保非負（與訓練時相同的處理）
        min_val = np.min(interaction_matrix_scaled)
        if min_val < 0:
            interaction_matrix_scaled = interaction_matrix_scaled - min_val
        interaction_matrix_scaled = np.maximum(interaction_matrix_scaled, 1e-8)
        
        nmf_features = self.nmf_model.transform(interaction_matrix_scaled)
        
        # 交叉驗證
        cv_scores = cross_val_score(
            self.classifier, nmf_features, y,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
            scoring=scoring
        )
        
        return {
            'cv_scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores)
        }
