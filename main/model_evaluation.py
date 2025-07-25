"""
模型訓練與評估模組
包含分類器訓練、性能評估和基準比較
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """模型訓練器"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def train_rule_based_model(self, rule_features, y, model_type='logistic'):
        """訓練基於規則的模型"""
        if rule_features.shape[1] == 0:
            print("警告：規則特徵為空，無法訓練模型")
            return None
        
        if model_type == 'logistic':
            model = LogisticRegression(random_state=self.random_state, max_iter=1000, n_jobs=-1)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(random_state=self.random_state, n_estimators=100, n_jobs=-1)
        elif model_type == 'decision_tree':
            model = DecisionTreeClassifier(random_state=self.random_state, max_depth=10, n_jobs=-1)
        else:
            raise ValueError(f"不支持的模型類型：{model_type}")
        
        model.fit(rule_features, y)
        self.models[f'rule_based_{model_type}'] = model
        
        print(f"規則模型({model_type})訓練完成，特徵數：{rule_features.shape[1]}")
        return model
    
    def train_baseline_models(self, X, y):
        """訓練基準模型"""
        baseline_models = {
            'logistic_regression': LogisticRegression(random_state=self.random_state, max_iter=1000, n_jobs=-1),
            'random_forest': RandomForestClassifier(random_state=self.random_state, n_estimators=100, n_jobs=-1),
            'decision_tree': DecisionTreeClassifier(random_state=self.random_state, max_depth=10),
            'svm': SVC(random_state=self.random_state, probability=True),
            'knn': KNeighborsClassifier(n_neighbors=5)
        }
        
        for name, model in baseline_models.items():
            model.fit(X, y)
            self.models[f'baseline_{name}'] = model
        
        print(f"訓練了 {len(baseline_models)} 個基準模型")
        return baseline_models


class ModelEvaluator:
    """模型評估器"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, model, X, y, model_name):
        """評估單個模型"""
        try:
            # 檢查是否為NMF互動項設計模型
            if hasattr(model, 'nmf_model') and hasattr(model, 'classifier'):
                # NMF模型需要特殊處理
                y_pred = model.predict(X)
                y_prob = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
            else:
                # 標準模型
                y_pred = model.predict(X)
                y_prob = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # 基本分類指標
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, zero_division=0),
                'recall': recall_score(y, y_pred, zero_division=0),
                'f1_score': f1_score(y, y_pred, zero_division=0),
            }
            
            # 機率預測指標
            if y_prob is not None:
                metrics['auc_roc'] = roc_auc_score(y, y_prob)
                metrics['brier_score'] = brier_score_loss(y, y_prob)
                metrics['log_loss'] = log_loss(y, y_prob)
            else:
                metrics['auc_roc'] = np.nan
                metrics['brier_score'] = np.nan
                metrics['log_loss'] = np.nan
            
            # 平衡準確率
            tn = np.sum((y == 0) & (y_pred == 0))
            tp = np.sum((y == 1) & (y_pred == 1))
            fn = np.sum((y == 1) & (y_pred == 0))
            fp = np.sum((y == 0) & (y_pred == 1))
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['balanced_accuracy'] = (sensitivity + specificity) / 2
            
            # 如果是NMF模型，添加額外的解釋性指標
            if hasattr(model, 'nmf_model') and hasattr(model, 'component_interpretations'):
                metrics['n_components'] = model.n_components
                metrics['reconstruction_error'] = getattr(model.nmf_model, 'reconstruction_err_', np.nan)
                metrics['n_interaction_features'] = model.interaction_matrix.shape[1] if hasattr(model, 'interaction_matrix') else np.nan
            
            self.results[model_name] = metrics
            
            return metrics
            
        except Exception as e:
            print(f"評估模型 {model_name} 時發生錯誤：{e}")
            return None
    
    def evaluate_all_models(self, models, X_rule, X_baseline, y):
        """評估所有模型"""
        all_results = {}
        
        # 評估規則模型
        for name, model in models.items():
            if 'rule_based' in name:
                if X_rule is not None and X_rule.shape[1] > 0:
                    result = self.evaluate_model(model, X_rule, y, name)
                    if result:
                        all_results[name] = result
            elif 'baseline' in name:
                result = self.evaluate_model(model, X_baseline, y, name)
                if result:
                    all_results[name] = result
        
        return all_results
    
    def cross_validate_model(self, model, X, y, cv=5, scoring='accuracy'):
        """交叉驗證評估"""
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            return {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'scores': scores
            }
        except Exception as e:
            print(f"交叉驗證失敗：{e}")
            return None
    
    def compute_calibration_metrics(self, y_true, y_prob, n_bins=10):
        """計算校準指標"""
        try:
            # 校準曲線
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=n_bins
            )
            
            # 校準誤差（ECE - Expected Calibration Error）
            ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            
            # 最大校準誤差（MCE - Maximum Calibration Error）
            mce = np.max(np.abs(fraction_of_positives - mean_predicted_value))
            
            return {
                'expected_calibration_error': ece,
                'maximum_calibration_error': mce,
                'fraction_of_positives': fraction_of_positives,
                'mean_predicted_value': mean_predicted_value
            }
        except Exception as e:
            print(f"計算校準指標失敗：{e}")
            return None
    
    def mcnemar_test(self, y_true, y_pred1, y_pred2):
        """McNemar檢定比較兩個模型"""
        # 構建混淆表
        correct1 = (y_true == y_pred1)
        correct2 = (y_true == y_pred2)
        
        # McNemar表
        both_correct = np.sum(correct1 & correct2)
        only_1_correct = np.sum(correct1 & ~correct2)
        only_2_correct = np.sum(~correct1 & correct2)
        both_wrong = np.sum(~correct1 & ~correct2)
        
        # McNemar統計量
        if only_1_correct + only_2_correct > 0:
            mcnemar_stat = (abs(only_1_correct - only_2_correct) - 1) ** 2 / (only_1_correct + only_2_correct)
        else:
            mcnemar_stat = 0
        
        # 使用卡方分布計算p值（自由度=1）
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(mcnemar_stat, df=1)
        
        return {
            'mcnemar_statistic': mcnemar_stat,
            'p_value': p_value,
            'contingency_table': {
                'both_correct': both_correct,
                'only_model1_correct': only_1_correct,
                'only_model2_correct': only_2_correct,
                'both_wrong': both_wrong
            }
        }
    
    def generate_performance_report(self, results):
        """生成性能報告"""
        if not results:
            return "沒有可用的評估結果"
        
        report = "模型性能評估報告\n" + "="*50 + "\n\n"
        
        # 按AUC排序
        sorted_models = sorted(results.items(), 
                             key=lambda x: x[1].get('auc_roc', 0), 
                             reverse=True)
        
        for model_name, metrics in sorted_models:
            report += f"{model_name}:\n"
            for metric, value in metrics.items():
                if isinstance(value, float):
                    report += f"  {metric}: {value:.4f}\n"
                else:
                    report += f"  {metric}: {value}\n"
            report += "\n"
        
        # 找出最佳模型
        best_model = sorted_models[0] if sorted_models else None
        if best_model:
            report += f"最佳模型（按AUC）: {best_model[0]}\n"
            report += f"AUC: {best_model[1].get('auc_roc', 'N/A'):.4f}\n"
            report += f"F1-Score: {best_model[1].get('f1_score', 'N/A'):.4f}\n"
        
        return report


if __name__ == "__main__":
    # 測試模型訓練和評估
    from data_preprocessing import load_and_preprocess_data
    from quantization_binning import QuantizationBinner
    from tensor_builder import TensorBuilder
    from cp_decomposition import NonNegativeCPDecomposition
    from rule_extraction import RuleExtractor
    
    X, y, feature_names = load_and_preprocess_data("dataset/UCI_BCD.csv")
    
    if X is not None:
        # 完整流程
        binner = QuantizationBinner(strategy='equal_width', n_bins=5)
        X_binned = binner.fit_transform(X, y, feature_names)
        
        builder = TensorBuilder(max_bins=6)
        tensor = builder.build_tensor(X_binned, method='one_hot')
        
        cp = NonNegativeCPDecomposition(rank=3, max_iter=30)
        cp.fit(tensor)
        
        extractor = RuleExtractor()
        rules = extractor.extract_rules(cp, feature_names, binner.get_bin_info())
        rule_features = extractor.generate_rule_features(X_binned)
        
        # 訓練模型
        trainer = ModelTrainer()
        
        # 規則模型
        if rule_features.shape[1] > 0:
            rule_model = trainer.train_rule_based_model(rule_features, y, 'logistic')
        
        # 基準模型
        baseline_models = trainer.train_baseline_models(X, y)
        
        # 評估模型
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_all_models(trainer.models, rule_features, X, y)
        
        # 生成報告
        report = evaluator.generate_performance_report(results)
        print(f"\n{report}")
