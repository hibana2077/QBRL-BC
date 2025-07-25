"""
決策曲線分析模組
實現臨床決策價值評估
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class DecisionCurveAnalysis:
    """決策曲線分析器"""
    
    def __init__(self):
        self.thresholds = np.linspace(0, 1, 101)
        self.results = {}
    
    def compute_net_benefit(self, y_true, y_prob, threshold):
        """
        計算淨效益
        
        Args:
            y_true: 真實標籤
            y_prob: 預測機率
            threshold: 決策閾值
        
        Returns:
            net_benefit: 淨效益值
        """
        # 檢查輸入數據
        if len(y_true) == 0 or len(y_prob) == 0:
            return 0.0
        
        # 檢查並處理 NaN 值
        if np.any(np.isnan(y_prob)) or np.any(np.isnan(y_true)):
            return 0.0
        
        # 確保 threshold 在合理範圍內
        threshold = np.clip(threshold, 0.0, 1.0)
        
        y_pred = (y_prob >= threshold).astype(int)
        
        # 混淆矩陣
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        except ValueError:
            # 如果只有一個類別，返回0
            return 0.0
        
        n = len(y_true)
        
        # 淨效益計算
        # Net Benefit = (TP/n) - (FP/n) * (pt/(1-pt))
        # 其中 pt 是閾值機率
        
        if threshold == 0:
            # 閾值為0時，所有人都治療
            net_benefit = np.mean(y_true)
        elif threshold >= 1.0:
            # 閾值為1時，沒有人治療
            net_benefit = 0.0
        else:
            odds = threshold / (1 - threshold)
            net_benefit = (tp / n) - (fp / n) * odds
        
        # 檢查結果是否為有效數值
        if np.isnan(net_benefit) or np.isinf(net_benefit):
            return 0.0
        
        return net_benefit
    
    def compute_decision_curve(self, y_true, y_prob, model_name="Model"):
        """計算決策曲線"""
        # 檢查輸入數據
        if len(y_true) == 0 or len(y_prob) == 0:
            print(f"警告：{model_name} 的輸入數據為空")
            return np.array([]), np.array([])
        
        # 檢查並處理 NaN 值
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_prob))
        if not np.any(valid_mask):
            print(f"警告：{model_name} 的所有數據都是 NaN")
            return np.array([]), np.array([])
        
        y_true_clean = y_true[valid_mask]
        y_prob_clean = y_prob[valid_mask]
        
        # 確保機率在 [0, 1] 範圍內
        y_prob_clean = np.clip(y_prob_clean, 0.0, 1.0)
        
        net_benefits = []
        
        for threshold in self.thresholds:
            nb = self.compute_net_benefit(y_true_clean, y_prob_clean, threshold)
            net_benefits.append(nb)
        
        net_benefits = np.array(net_benefits)
        
        # 檢查計算結果
        if np.any(np.isnan(net_benefits)) or np.any(np.isinf(net_benefits)):
            print(f"警告：{model_name} 的淨效益計算包含 NaN 或無窮大值，已清理")
            net_benefits = np.nan_to_num(net_benefits, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.results[model_name] = {
            'thresholds': self.thresholds,
            'net_benefits': net_benefits
        }
        
        return self.thresholds, net_benefits
    
    def compute_reference_strategies(self, y_true):
        """計算參考策略的淨效益"""
        # 檢查輸入數據
        if len(y_true) == 0:
            print("警告：參考策略的輸入數據為空")
            return np.array([]), np.array([]), np.array([])
        
        # 檢查並處理 NaN 值
        valid_mask = ~np.isnan(y_true)
        if not np.any(valid_mask):
            print("警告：參考策略的所有數據都是 NaN")
            return np.array([]), np.array([]), np.array([])
        
        y_true_clean = y_true[valid_mask]
        prevalence = np.mean(y_true_clean)
        
        # 治療所有人的策略
        treat_all_benefits = []
        for threshold in self.thresholds:
            if threshold == 0:
                nb = prevalence
            elif threshold >= 1.0:
                nb = 0.0
            else:
                odds = threshold / (1 - threshold)
                nb = prevalence - odds
            
            # 檢查結果是否為有效數值
            if np.isnan(nb) or np.isinf(nb):
                nb = 0.0
            
            treat_all_benefits.append(nb)
        
        # 不治療任何人的策略（淨效益恆為0）
        treat_none_benefits = np.zeros_like(self.thresholds)
        
        treat_all_benefits = np.array(treat_all_benefits)
        
        self.results['Treat All'] = {
            'thresholds': self.thresholds,
            'net_benefits': treat_all_benefits
        }
        
        self.results['Treat None'] = {
            'thresholds': self.thresholds,
            'net_benefits': treat_none_benefits
        }
        
        return self.thresholds, treat_all_benefits, treat_none_benefits
    
    def plot_decision_curves(self, save_path=None, figsize=(10, 6)):
        """繪製決策曲線"""
        if not self.results:
            print("沒有可繪製的決策曲線數據")
            return
        
        plt.figure(figsize=figsize)
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
        color_idx = 0
        
        valid_results = {}
        
        for model_name, data in self.results.items():
            thresholds = data['thresholds']
            net_benefits = data['net_benefits']
            
            # 檢查數據有效性
            if len(thresholds) == 0 or len(net_benefits) == 0:
                print(f"跳過 {model_name}：數據為空")
                continue
            
            # 清理 NaN 和無窮大值
            valid_mask = np.isfinite(net_benefits)
            if not np.any(valid_mask):
                print(f"跳過 {model_name}：所有數據都無效")
                continue
            
            # 使用有效數據
            valid_thresholds = thresholds[valid_mask]
            valid_net_benefits = net_benefits[valid_mask]
            
            valid_results[model_name] = {
                'thresholds': valid_thresholds,
                'net_benefits': valid_net_benefits
            }
            
            if model_name in ['Treat All', 'Treat None']:
                # 參考策略使用虛線
                linestyle = '--'
                alpha = 0.7
            else:
                # 模型使用實線
                linestyle = '-'
                alpha = 1.0
            
            plt.plot(valid_thresholds, valid_net_benefits, 
                    label=model_name, 
                    linestyle=linestyle, 
                    alpha=alpha,
                    color=colors[color_idx % len(colors)],
                    linewidth=2)
            color_idx += 1
        
        if not valid_results:
            print("沒有有效的決策曲線數據可繪製")
            plt.close()
            return
        
        plt.xlabel('Decision Threshold', fontsize=12)
        plt.ylabel('Net Benefit', fontsize=12)
        plt.title('Decision Curve Analysis', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 設定合理的y軸範圍
        try:
            all_benefits = np.concatenate([data['net_benefits'] for data in valid_results.values()])
            if len(all_benefits) > 0 and np.any(np.isfinite(all_benefits)):
                finite_benefits = all_benefits[np.isfinite(all_benefits)]
                max_benefit = np.max(finite_benefits)
                min_benefit = np.min(finite_benefits)
                y_range = max_benefit - min_benefit
                
                if y_range > 0:
                    plt.ylim(min_benefit - 0.1 * y_range, max_benefit + 0.1 * y_range)
                else:
                    plt.ylim(min_benefit - 0.1, max_benefit + 0.1)
            else:
                plt.ylim(-0.1, 0.1)
        except Exception as e:
            print(f"設定 y 軸範圍時出錯：{e}")
            plt.ylim(-0.1, 0.1)
        
        plt.tight_layout()
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"決策曲線圖已保存至：{save_path}")
            except Exception as e:
                print(f"保存決策曲線圖失敗：{e}")
        
        try:
            plt.show()
        except Exception as e:
            print(f"顯示決策曲線圖失敗：{e}")
            
        plt.close()
    
    def find_optimal_threshold(self, model_name):
        """找到最佳決策閾值"""
        if model_name not in self.results:
            print(f"模型 {model_name} 不存在")
            return None
        
        data = self.results[model_name]
        net_benefits = data['net_benefits']
        thresholds = data['thresholds']
        
        # 檢查數據有效性
        if len(net_benefits) == 0:
            print(f"模型 {model_name} 沒有有效的淨效益數據")
            return None
        
        # 只考慮有限的數值
        valid_mask = np.isfinite(net_benefits)
        if not np.any(valid_mask):
            print(f"模型 {model_name} 沒有有效的淨效益數據")
            return None
        
        valid_benefits = net_benefits[valid_mask]
        valid_thresholds = thresholds[valid_mask]
        
        # 找到淨效益最大的閾值
        max_idx = np.argmax(valid_benefits)
        optimal_threshold = valid_thresholds[max_idx]
        max_net_benefit = valid_benefits[max_idx]
        
        return {
            'optimal_threshold': optimal_threshold,
            'max_net_benefit': max_net_benefit,
            'threshold_index': max_idx
        }
    
    def compare_models_at_threshold(self, threshold):
        """在特定閾值下比較所有模型"""
        if abs(threshold - 1.0) < 1e-10:
            threshold_idx = -1
        else:
            threshold_idx = np.argmin(np.abs(self.thresholds - threshold))
        
        comparison = {}
        
        for model_name, data in self.results.items():
            if model_name not in ['Treat All', 'Treat None']:  # 排除參考策略
                net_benefit = data['net_benefits'][threshold_idx]
                comparison[model_name] = net_benefit
        
        # 排序
        sorted_comparison = sorted(comparison.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n在閾值 {threshold:.2f} 下的模型比較：")
        for i, (model_name, net_benefit) in enumerate(sorted_comparison):
            print(f"{i+1}. {model_name}: {net_benefit:.4f}")
        
        return sorted_comparison
    
    def compute_clinical_impact(self, y_true, y_prob, threshold, population_size=1000):
        """
        計算臨床影響
        
        Args:
            y_true: 真實標籤
            y_prob: 預測機率
            threshold: 決策閾值
            population_size: 目標人群大小
        """
        y_pred = (y_prob >= threshold).astype(int)
        
        # 混淆矩陣
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        n = len(y_true)
        
        # 按比例推算到目標人群
        scale_factor = population_size / n
        
        impact = {
            'threshold': threshold,
            'population_size': population_size,
            'predicted_prevalence': np.mean(y_true),
            'model_identified_high_risk': np.sum(y_pred) * scale_factor,
            'true_positives': tp * scale_factor,
            'false_positives': fp * scale_factor,
            'true_negatives': tn * scale_factor,
            'false_negatives': fn * scale_factor,
            'number_needed_to_treat': (tp + fp) / tp if tp > 0 else float('inf'),
            'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0
        }
        
        return impact
    
    def generate_dca_report(self, y_true, y_prob, model_name, population_size=1000):
        """生成DCA分析報告"""
        # 計算決策曲線
        thresholds, net_benefits = self.compute_decision_curve(y_true, y_prob, model_name)
        
        # 找到最佳閾值
        optimal = self.find_optimal_threshold(model_name)
        
        # 計算臨床影響
        clinical_impact = self.compute_clinical_impact(
            y_true, y_prob, optimal['optimal_threshold'], population_size
        )
        
        report = f"決策曲線分析報告 - {model_name}\n"
        report += "=" * 50 + "\n\n"
        
        report += f"最佳決策閾值: {optimal['optimal_threshold']:.3f}\n"
        report += f"最大淨效益: {optimal['max_net_benefit']:.4f}\n\n"
        
        report += f"臨床影響評估 (人群規模: {population_size}):\n"
        report += f"  預測高風險人數: {clinical_impact['model_identified_high_risk']:.0f}\n"
        report += f"  真陽性: {clinical_impact['true_positives']:.0f}\n"
        report += f"  假陽性: {clinical_impact['false_positives']:.0f}\n"
        report += f"  治療所需數量 (NNT): {clinical_impact['number_needed_to_treat']:.1f}\n"
        report += f"  陽性預測值: {clinical_impact['positive_predictive_value']:.3f}\n"
        report += f"  陰性預測值: {clinical_impact['negative_predictive_value']:.3f}\n"
        
        return report


if __name__ == "__main__":
    # 測試決策曲線分析
    from data_preprocessing import load_and_preprocess_data
    from model_evaluation import ModelTrainer, ModelEvaluator
    
    X, y, feature_names = load_and_preprocess_data("dataset/UCI_BCD.csv")
    
    if X is not None:
        # 訓練一個簡單的基準模型
        trainer = ModelTrainer()
        baseline_models = trainer.train_baseline_models(X, y)
        
        # 獲取預測機率
        model = trainer.models['baseline_logistic_regression']
        y_prob = model.predict_proba(X)[:, 1]
        
        # 決策曲線分析
        dca = DecisionCurveAnalysis()
        
        # 計算模型和參考策略的決策曲線
        dca.compute_decision_curve(y, y_prob, "Logistic Regression")
        dca.compute_reference_strategies(y)
        
        # 生成報告
        report = dca.generate_dca_report(y, y_prob, "Logistic Regression")
        print(report)
        
        # 比較不同閾值下的性能
        dca.compare_models_at_threshold(0.5)
