"""
規則提取與特徵生成模組
將CP分解因子轉換為可解釋的規則和新特徵
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class RuleExtractor:
    """規則提取器"""
    
    def __init__(self, feature_threshold=0.1, bin_threshold=0.1, min_support=0.05):
        """
        初始化規則提取器
        
        Args:
            feature_threshold: 特徵權重閾值
            bin_threshold: 分箱權重閾值  
            min_support: 最小支持度
        """
        self.feature_threshold = feature_threshold
        self.bin_threshold = bin_threshold
        self.min_support = min_support
        
        self.rules_ = []
        self.feature_names = None
        self.bin_info = None
        
    def extract_rules(self, cp_model, feature_names, bin_info):
        """
        從CP分解模型提取規則
        
        Args:
            cp_model: 已訓練的CP分解模型
            feature_names: 特徵名稱列表
            bin_info: 分箱信息字典
        
        Returns:
            rules: 提取的規則列表
        """
        self.feature_names = feature_names
        self.bin_info = bin_info
        
        if cp_model.factors_ is None:
            raise ValueError("CP模型尚未訓練")
        
        A, B, C = cp_model.factors_
        weights = cp_model.weights_
        
        rules = []
        
        for r in range(cp_model.rank):
            rule = self._extract_single_rule(r, B[:, r], C[:, r], weights[r])
            if rule['conditions']:  # 只保留有條件的規則
                rules.append(rule)
        
        self.rules_ = rules
        print(f"提取了 {len(rules)} 條規則")
        
        return rules
    
    def _extract_single_rule(self, rule_id, feature_weights, bin_weights, factor_weight):
        """提取單個規則"""
        rule = {
            'rule_id': rule_id,
            'factor_weight': factor_weight,
            'conditions': [],
            'support': 0.0,
            'confidence': 0.0
        }
        
        # 找到重要的特徵
        significant_features = np.where(feature_weights > self.feature_threshold)[0]
        
        for feat_idx in significant_features:
            feature_name = self.feature_names[feat_idx]
            feature_weight = feature_weights[feat_idx]
            
            # 找到該特徵的重要分箱
            significant_bins = np.where(bin_weights > self.bin_threshold)[0]
            
            if len(significant_bins) > 0 and feature_name in self.bin_info:
                # 取權重最高的分箱
                best_bin_idx = significant_bins[np.argmax(bin_weights[significant_bins])]
                
                if best_bin_idx < len(self.bin_info[feature_name]['ranges']):
                    bin_range = self.bin_info[feature_name]['ranges'][best_bin_idx]
                    
                    condition = {
                        'feature': feature_name,
                        'feature_weight': feature_weight,
                        'bin_idx': int(best_bin_idx),
                        'range': bin_range,
                        'bin_weight': bin_weights[best_bin_idx]
                    }
                    
                    rule['conditions'].append(condition)
        
        return rule
    
    def generate_rule_features(self, X_binned, tensor=None):
        """
        生成基於規則的新特徵
        
        Args:
            X_binned: 分箱後的數據
            tensor: 原始張量（可選）
        
        Returns:
            rule_features: 規則特徵矩陣
        """
        if not self.rules_:
            raise ValueError("尚未提取規則，請先調用extract_rules方法")
        
        n_samples = X_binned.shape[0]
        n_rules = len(self.rules_)
        
        rule_features = np.zeros((n_samples, n_rules))
        
        for i, rule in enumerate(self.rules_):
            # 計算每個樣本對該規則的激活程度
            activation = self._compute_rule_activation(X_binned, rule)
            rule_features[:, i] = activation
        
        # 計算規則統計
        self._compute_rule_statistics(X_binned, rule_features)
        
        print(f"生成了 {n_rules} 個規則特徵")
        return rule_features
    
    def _compute_rule_activation(self, X_binned, rule):
        """計算規則激活程度"""
        n_samples = X_binned.shape[0]
        activation = np.zeros(n_samples)
        
        if not rule['conditions']:
            return activation
        
        for i in range(n_samples):
            sample_activation = 1.0
            
            for condition in rule['conditions']:
                feature_name = condition['feature']
                target_bin = condition['bin_idx']
                
                # 找到特徵索引
                feature_idx = None
                for j, fname in enumerate(self.feature_names):
                    if fname == feature_name:
                        feature_idx = j
                        break
                
                if feature_idx is not None:
                    sample_bin = X_binned[i, feature_idx]
                    if sample_bin == target_bin:
                        # 樣本滿足條件，使用條件權重
                        sample_activation *= condition['feature_weight'] * condition['bin_weight']
                    else:
                        # 樣本不滿足條件
                        sample_activation = 0.0
                        break
            
            activation[i] = sample_activation
        
        return activation
    
    def _compute_rule_statistics(self, X_binned, rule_features):
        """計算規則統計信息"""
        n_samples = X_binned.shape[0]
        
        for i, rule in enumerate(self.rules_):
            activations = rule_features[:, i]
            
            # 支持度：激活該規則的樣本比例
            support = np.mean(activations > 0)
            rule['support'] = support
            
            # 平均激活強度
            rule['avg_activation'] = np.mean(activations[activations > 0]) if support > 0 else 0.0
            
            # 激活樣本數
            rule['activated_samples'] = np.sum(activations > 0)
    
    def get_rule_descriptions(self, max_conditions=3):
        """獲取規則的自然語言描述"""
        descriptions = []
        
        for rule in self.rules_:
            desc = f"規則 {rule['rule_id']} (權重: {rule['factor_weight']:.3f}, 支持度: {rule['support']:.3f}):\n"
            
            if rule['conditions']:
                desc += "  如果: "
                condition_strs = []
                
                for j, condition in enumerate(rule['conditions'][:max_conditions]):
                    feature = condition['feature']
                    range_min, range_max = condition['range']
                    condition_str = f"{feature} ∈ [{range_min:.3f}, {range_max:.3f}]"
                    condition_strs.append(condition_str)
                
                desc += " AND ".join(condition_strs)
                
                if len(rule['conditions']) > max_conditions:
                    desc += f" ... (還有 {len(rule['conditions']) - max_conditions} 個條件)"
                
                desc += f"\n  則: 因子激活 (平均強度: {rule.get('avg_activation', 0):.3f})"
            else:
                desc += "  無有效條件"
            
            descriptions.append(desc)
        
        return descriptions
    
    def evaluate_rule_quality(self, rule_features, y):
        """評估規則品質"""
        if len(self.rules_) == 0:
            return {}
        
        from sklearn.metrics import mutual_info_score
        
        quality_metrics = {}
        
        for i, rule in enumerate(self.rules_):
            rule_activation = rule_features[:, i] > 0
            
            # 計算與標籤的互信息
            if np.sum(rule_activation) > 1:  # 至少有2個樣本激活
                mi_score = mutual_info_score(y, rule_activation)
                
                # 計算正負樣本的激活率
                pos_activation = np.mean(rule_activation[y == 1])
                neg_activation = np.mean(rule_activation[y == 0])
                
                quality_metrics[f'rule_{i}'] = {
                    'mutual_info': mi_score,
                    'pos_activation_rate': pos_activation,
                    'neg_activation_rate': neg_activation,
                    'discrimination': abs(pos_activation - neg_activation),
                    'support': rule['support']
                }
        
        return quality_metrics
    
    def filter_rules(self, rule_features, y, min_discrimination=0.1, min_mutual_info=0.01):
        """過濾低品質規則"""
        quality_metrics = self.evaluate_rule_quality(rule_features, y)
        
        good_rules = []
        good_features = []
        
        for i, rule in enumerate(self.rules_):
            rule_key = f'rule_{i}'
            if rule_key in quality_metrics:
                metrics = quality_metrics[rule_key]
                
                if (metrics['discrimination'] >= min_discrimination and 
                    metrics['mutual_info'] >= min_mutual_info):
                    good_rules.append(rule)
                    good_features.append(rule_features[:, i])
        
        if good_features:
            filtered_features = np.column_stack(good_features)
            print(f"過濾後保留 {len(good_rules)} 個高品質規則")
            return good_rules, filtered_features
        else:
            print("沒有符合條件的高品質規則")
            return [], np.array([]).reshape(len(y), 0)


if __name__ == "__main__":
    # 測試規則提取
    from data_preprocessing import load_and_preprocess_data
    from quantization_binning import QuantizationBinner
    from tensor_builder import TensorBuilder
    from cp_decomposition import NonNegativeCPDecomposition
    
    X, y, feature_names = load_and_preprocess_data("dataset/UCI_BCD.csv")
    
    if X is not None:
        # 完整流程測試
        binner = QuantizationBinner(strategy='equal_width', n_bins=5)
        X_binned = binner.fit_transform(X, y, feature_names)
        
        builder = TensorBuilder(max_bins=6)
        tensor = builder.build_tensor(X_binned, method='one_hot')
        
        cp = NonNegativeCPDecomposition(rank=3, max_iter=30)
        cp.fit(tensor)
        
        # 提取規則
        extractor = RuleExtractor()
        rules = extractor.extract_rules(cp, feature_names, binner.get_bin_info())
        
        # 生成規則特徵
        rule_features = extractor.generate_rule_features(X_binned)
        
        # 獲取規則描述
        descriptions = extractor.get_rule_descriptions()
        for desc in descriptions[:2]:  # 顯示前2個規則
            print(f"\n{desc}")
        
        # 評估規則品質
        quality = extractor.evaluate_rule_quality(rule_features, y)
        print(f"\n規則品質評估：")
        for rule_key, metrics in quality.items():
            print(f"{rule_key}: 互信息={metrics['mutual_info']:.3f}, 區分度={metrics['discrimination']:.3f}")
