"""
實驗管道主程式
整合所有模組，執行完整的乳癌診斷模型實驗
"""

import numpy as np
import pandas as pd
import os
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 導入自定義模組
from data_preprocessing import DataPreprocessor
from quantization_binning import QuantizationBinner
from tensor_builder import TensorBuilder
from cp_decomposition import NonNegativeCPDecomposition
from rule_extraction import RuleExtractor
from model_evaluation import ModelTrainer, ModelEvaluator
from decision_curve_analysis import DecisionCurveAnalysis
from nmf_interaction_design import NMFInteractionDesign


class ExperimentPipeline:
    """實驗管道類別"""
    
    def __init__(self, data_path, output_dir="results", random_state=42):
        """
        初始化實驗管道
        
        Args:
            data_path: 資料路徑
            output_dir: 結果輸出目錄
            random_state: 隨機種子
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.random_state = random_state
        
        # 創建輸出目錄
        os.makedirs(output_dir, exist_ok=True)
        
        # 實驗設定
        self.config = {
            'binning_strategies': ['equal_width', 'equal_frequency', 'mdlp'],
            'n_bins_options': [3, 5, 7],
            'tensor_methods': ['one_hot', 'weighted'],
            'cp_ranks': [3, 4],
            'max_cp_iter': 50,
            # NMF互動項設計參數
            'nmf_components': [12, 13],
            'use_nmf_fallback': True,  # 當CP分解效果不佳時使用NMF
            'nmf_alpha': [0.01]
        }
        
        # 結果存儲
        self.results = {
            'data_info': {},
            'experiments': [],
            'best_config': None,
            'summary': {}
        }
        
    def run_full_experiment(self):
        """執行完整實驗"""
        print("=" * 60)
        print("QBRL-BC 乳癌診斷模型實驗開始")
        print("=" * 60)
        
        start_time = time.time()
        
        # 1. 資料載入與預處理
        print("\n1. 資料載入與預處理...")
        X, y, feature_names = self._load_and_prepare_data()
        
        if X is None:
            print("資料載入失敗，實驗終止")
            return None
        
        # 2. 參數搜索實驗
        print("\n2. 開始參數搜索實驗...")
        best_config = self._parameter_search(X, y, feature_names)
        
        # 3. 使用最佳配置進行完整評估
        print("\n3. 使用最佳配置進行完整評估...")
        final_results = self._final_evaluation(X, y, feature_names, best_config)
        
        # 4. 決策曲線分析
        print("\n4. 執行決策曲線分析...")
        dca_results = self._decision_curve_analysis(X, y, feature_names, best_config)
        
        # 5. 保存結果
        print("\n5. 保存實驗結果...")
        self._save_results(final_results, dca_results)
        
        total_time = time.time() - start_time
        print(f"\n實驗完成！總耗時：{total_time:.2f} 秒")
        
        return self.results
    
    def _load_and_prepare_data(self):
        """載入並準備資料"""
        preprocessor = DataPreprocessor(self.data_path)
        X, y = preprocessor.load_data()
        
        if X is not None and y is not None:
            # 檢查資料品質
            is_clean = preprocessor.check_data_quality(X, y)
            
            if is_clean:
                # 標準化特徵
                X_scaled = preprocessor.standardize_features(X)
                
                # 保存資料信息
                self.results['data_info'] = {
                    'n_samples': X.shape[0],
                    'n_features': X.shape[1],
                    'n_positive': int(np.sum(y)),
                    'n_negative': int(np.sum(1-y)),
                    'feature_names': preprocessor.feature_names
                }
                
                return X_scaled, y, preprocessor.feature_names
        
        return None, None, None
    
    def _parameter_search(self, X, y, feature_names):
        """參數搜索"""
        best_score = -1
        best_config = None
        experiment_count = 0
        
        # 計算總實驗數（包括NMF參數）
        total_experiments = (len(self.config['binning_strategies']) * 
                           len(self.config['n_bins_options']) * 
                           len(self.config['tensor_methods']) * 
                           len(self.config['cp_ranks']) *
                           len(self.config['nmf_components']) *
                           len(self.config['nmf_alpha'])
                           )

        print(f"總共需要執行 {total_experiments} 個CP分解實驗組合")
        print("如果CP分解效果不佳，將自動嘗試NMF + 互動項設計方法")
        
        for binning_strategy in self.config['binning_strategies']:
            for n_bins in self.config['n_bins_options']:
                for tensor_method in self.config['tensor_methods']:
                    for cp_rank in self.config['cp_ranks']:
                        for nmf_components in self.config['nmf_components']:
                            for nmf_alpha in self.config['nmf_alpha']:
                        
                                experiment_count += 1
                                print(f"\n實驗 {experiment_count}/{total_experiments}: "
                                    f"{binning_strategy}, {n_bins}bins, {tensor_method}, rank={cp_rank}, "
                                    f"nmf_components={nmf_components}, nmf_alpha={nmf_alpha}")

                                try:
                                    # 執行單個實驗（包含CP和NMF方法）
                                    config = {
                                        'binning_strategy': binning_strategy,
                                        'n_bins': n_bins,
                                        'tensor_method': tensor_method,
                                        'cp_rank': cp_rank,
                                        'nmf_components': nmf_components,
                                        'nmf_alpha': nmf_alpha
                                    }
                                    
                                    score = self._run_single_experiment(X, y, feature_names, config)
                                    
                                    if score > best_score:
                                        best_score = score
                                        best_config = config.copy()
                                        print(f"  新的最佳配置！分數: {best_score:.4f}")
                                    
                                except Exception as e:
                                    print(f"  實驗失敗：{e}")
                                    continue
        
        # 如果最佳分數仍然不理想，專門嘗試NMF參數調優
        if best_score < 0.75:
            print(f"\n最佳CP分解分數 {best_score:.4f} 不理想，進行NMF參數調優...")
            best_config, best_score = self._nmf_parameter_search(X, y, feature_names, best_config, best_score)
        
        print(f"\n參數搜索完成，最佳配置：{best_config}")
        print(f"最佳分數：{best_score:.4f}")
        
        self.results['best_config'] = best_config
        return best_config
    
    def _nmf_parameter_search(self, X, y, feature_names, current_best_config, current_best_score):
        """NMF專用參數搜索"""
        print("執行NMF + 互動項設計參數調優...")
        
        best_config = current_best_config
        best_score = current_best_score
        
        # 使用最佳的分箱配置
        best_binning = best_config['binning_strategy']
        best_n_bins = best_config['n_bins']
        
        nmf_experiment_count = 0
        total_nmf_experiments = len(self.config['nmf_components']) * len(self.config['nmf_alpha'])
        
        for n_components in self.config['nmf_components']:
            for alpha in self.config['nmf_alpha']:
                nmf_experiment_count += 1
                print(f"  NMF實驗 {nmf_experiment_count}/{total_nmf_experiments}: "
                      f"components={n_components}, alpha={alpha}")
                
                try:
                    config = {
                        'binning_strategy': best_binning,
                        'n_bins': best_n_bins,
                        'nmf_components': n_components,
                        'nmf_alpha': alpha,
                        'method': 'nmf_only'  # 標記為純NMF方法
                    }
                    
                    # 分箱
                    binner = QuantizationBinner(
                        strategy=config['binning_strategy'],
                        n_bins=config['n_bins']
                    )
                    X_binned = binner.fit_transform(X, y, feature_names)
                    
                    # 直接使用NMF方法
                    score = self._try_nmf_interaction(X_binned, y, feature_names, config, binner)
                    
                    if score > best_score:
                        best_score = score
                        best_config = config.copy()
                        print(f"    新的最佳NMF配置！分數: {best_score:.4f}")
                
                except Exception as e:
                    print(f"    NMF實驗失敗：{e}")
                    continue
        
        return best_config, best_score
    
    def _run_single_experiment(self, X, y, feature_names, config):
        """執行單個實驗"""
        # 分箱
        binner = QuantizationBinner(
            strategy=config['binning_strategy'],
            n_bins=config['n_bins']
        )
        X_binned = binner.fit_transform(X, y, feature_names)
        
        # 嘗試CP分解方法
        cp_score = self._try_cp_decomposition(X_binned, y, feature_names, config, binner)
        
        # 如果CP分解效果不佳，嘗試NMF方法
        if cp_score < 0.6 and self.config.get('use_nmf_fallback', True):
            print("    CP分解效果不佳，嘗試NMF + 互動項設計...")
            nmf_score = self._try_nmf_interaction(X_binned, y, feature_names, config, binner)
            return max(cp_score, nmf_score)
        
        return cp_score
    
    def _try_cp_decomposition(self, X_binned, y, feature_names, config, binner):
        """嘗試CP分解方法"""
        try:
            # 構建張量
            builder = TensorBuilder(max_bins=config['n_bins'] + 2)
            tensor = builder.build_tensor(X_binned, method=config['tensor_method'])
            
            # CP分解
            cp = NonNegativeCPDecomposition(
                rank=config['cp_rank'],
                max_iter=self.config['max_cp_iter'],
                random_state=self.random_state
            )
            cp.fit(tensor)
            
            # 規則提取
            extractor = RuleExtractor()
            rules = extractor.extract_rules(cp, feature_names, binner.get_bin_info())
            rule_features = extractor.generate_rule_features(X_binned)
            
            if rule_features.shape[1] == 0:
                return 0.0  # 沒有有效規則
            
            # 過濾規則
            good_rules, filtered_features = extractor.filter_rules(rule_features, y)
            
            if filtered_features.shape[1] == 0:
                return 0.0  # 沒有高品質規則
            
            # 訓練模型
            trainer = ModelTrainer(random_state=self.random_state)
            model = trainer.train_rule_based_model(filtered_features, y, 'logistic')
            
            # 評估
            evaluator = ModelEvaluator()
            result = evaluator.evaluate_model(model, filtered_features, y, 'rule_model')
            
            # 返回AUC作為評估分數
            return result.get('auc_roc', 0.0) if result else 0.0
            
        except Exception as e:
            print(f"    CP分解失敗: {e}")
            return 0.0
    
    def _try_nmf_interaction(self, X_binned, y, feature_names, config, binner):
        """嘗試NMF + 互動項設計方法"""
        try:
            # 使用配置中的NMF參數，如果沒有則使用默認值
            n_components = config.get('nmf_components', 10)
            alpha = config.get('nmf_alpha', 0.1)
            
            # 創建NMF模型
            nmf_model = NMFInteractionDesign(
                n_components=n_components,
                max_iter=200,
                random_state=self.random_state,
                alpha=alpha
            )
            
            # 訓練模型
            nmf_model.fit(
                X_binned, y, 
                feature_names=feature_names,
                bin_info=binner.get_bin_info(),
                add_interactions=True,
                classifier_type='logistic'
            )
            
            # 交叉驗證評估
            cv_result = nmf_model.cross_validate(X_binned, y, cv=3, scoring='roc_auc')
            
            return cv_result['mean_score']
            
        except Exception as e:
            print(f"    NMF方法失敗: {e}")
            return 0.0
    
    def _final_evaluation(self, X, y, feature_names, best_config):
        """使用最佳配置進行完整評估"""
        print("執行完整評估流程...")
        
        # 使用最佳配置重新訓練
        binner = QuantizationBinner(
            strategy=best_config['binning_strategy'],
            n_bins=best_config['n_bins']
        )
        X_binned = binner.fit_transform(X, y, feature_names)
        
        # 檢查是否使用純NMF方法
        if best_config.get('method') == 'nmf_only':
            return self._final_evaluation_nmf(X, X_binned, y, feature_names, best_config, binner)
        else:
            return self._final_evaluation_cp(X, X_binned, y, feature_names, best_config, binner)
    
    def _final_evaluation_cp(self, X, X_binned, y, feature_names, best_config, binner):
        """CP分解方法的最終評估"""
        builder = TensorBuilder(max_bins=best_config['n_bins'] + 2)
        tensor = builder.build_tensor(X_binned, method=best_config['tensor_method'])
        
        cp = NonNegativeCPDecomposition(
            rank=best_config['cp_rank'],
            max_iter=self.config['max_cp_iter'],
            random_state=self.random_state
        )
        cp.fit(tensor)
        
        extractor = RuleExtractor()
        rules = extractor.extract_rules(cp, feature_names, binner.get_bin_info())
        rule_features = extractor.generate_rule_features(X_binned)
        good_rules, filtered_features = extractor.filter_rules(rule_features, y)
        
        # 訓練所有模型
        trainer = ModelTrainer(random_state=self.random_state)
        
        # 規則模型
        if filtered_features.shape[1] > 0:
            trainer.train_rule_based_model(filtered_features, y, 'logistic')
            trainer.train_rule_based_model(filtered_features, y, 'random_forest')
        else:
            # 如果沒有高品質規則，使用 CP 因子作為特徵
            print("警告：沒有高品質規則，使用 CP 因子作為特徵訓練模型")
            cp_factors = cp.transform(tensor)
            if cp_factors is not None and cp_factors.shape[1] > 0:
                trainer.train_rule_based_model(cp_factors, y, 'logistic')
                trainer.train_rule_based_model(cp_factors, y, 'random_forest')
                # 更新 filtered_features 為 CP 因子
                filtered_features = cp_factors
        
        # 基準模型
        trainer.train_baseline_models(X, y)
        
        # 評估所有模型
        evaluator = ModelEvaluator()
        all_results = evaluator.evaluate_all_models(trainer.models, filtered_features, X, y)
        
        # 生成報告
        report = evaluator.generate_performance_report(all_results)
        print(report)
        
        # 保存詳細結果
        final_results = {
            'method': 'CP_Decomposition',
            'config': best_config,
            'n_rules': len(good_rules),
            'n_rule_features': filtered_features.shape[1],
            'model_results': all_results,
            'rules': [
                {
                    'rule_id': rule['rule_id'],
                    'weight': rule['factor_weight'],
                    'support': rule['support'],
                    'n_conditions': len(rule['conditions'])
                }
                for rule in good_rules
            ],
            'rule_descriptions': extractor.get_rule_descriptions()
        }
        
        return final_results
    
    def _final_evaluation_nmf(self, X, X_binned, y, feature_names, best_config, binner):
        """NMF方法的最終評估"""
        print("使用NMF + 互動項設計進行最終評估...")
        
        # 創建NMF模型
        nmf_model = NMFInteractionDesign(
            n_components=best_config['nmf_components'],
            max_iter=200,
            random_state=self.random_state,
            alpha=best_config['nmf_alpha']
        )
        
        # 訓練NMF模型
        nmf_model.fit(
            X_binned, y, 
            feature_names=feature_names,
            bin_info=binner.get_bin_info(),
            add_interactions=True,
            classifier_type='logistic'
        )
        
        # 獲取NMF特徵
        nmf_features = nmf_model.nmf_features
        
        # 訓練基準模型
        trainer = ModelTrainer(random_state=self.random_state)
        trainer.train_baseline_models(X, y)
        
        # 添加NMF模型到比較
        trainer.models['NMF_Interaction'] = nmf_model
        
        # 評估所有模型
        evaluator = ModelEvaluator()
        all_results = {}
        
        # 評估基準模型
        for model_name, model in trainer.models.items():
            if model_name != 'NMF_Interaction':
                result = evaluator.evaluate_model(model, X, y, model_name)
                all_results[model_name] = result
        
        # 評估NMF模型
        nmf_result = evaluator.evaluate_model(nmf_model, X_binned, y, 'NMF_Interaction')
        all_results['NMF_Interaction'] = nmf_result
        
        # 生成報告
        report = evaluator.generate_performance_report(all_results)
        print(report)
        
        # 獲取NMF模型摘要
        model_summary = nmf_model.get_model_summary()
        
        # 保存詳細結果
        final_results = {
            'method': 'NMF_Interaction_Design',
            'config': best_config,
            'n_components': best_config['nmf_components'],
            'n_interaction_features': nmf_model.interaction_matrix.shape[1],
            'nmf_reconstruction_error': model_summary.get('reconstruction_error'),
            'model_results': all_results,
            'component_interpretations': model_summary.get('component_interpretations', []),
            'nmf_model_summary': model_summary
        }
        
        return final_results
    
    def _decision_curve_analysis(self, X, y, feature_names, best_config):
        """決策曲線分析"""
        print("執行決策曲線分析...")
        
        try:
            # 重新訓練最佳模型
            binner = QuantizationBinner(
                strategy=best_config['binning_strategy'],
                n_bins=best_config['n_bins']
            )
            X_binned = binner.fit_transform(X, y, feature_names)
            
            builder = TensorBuilder(max_bins=best_config['n_bins'] + 2)
            tensor = builder.build_tensor(X_binned, method=best_config['tensor_method'])
            
            cp = NonNegativeCPDecomposition(
                rank=best_config['cp_rank'],
                max_iter=self.config['max_cp_iter'],
                random_state=self.random_state
            )
            cp.fit(tensor)
            
            extractor = RuleExtractor()
            rules = extractor.extract_rules(cp, feature_names, binner.get_bin_info())
            rule_features = extractor.generate_rule_features(X_binned)
            good_rules, filtered_features = extractor.filter_rules(rule_features, y)
            
            # 訓練模型
            trainer = ModelTrainer(random_state=self.random_state)
            
            dca = DecisionCurveAnalysis()
            dca_results = {}
            
            # 規則模型DCA
            if filtered_features.shape[1] > 0:
                try:
                    rule_model = trainer.train_rule_based_model(filtered_features, y, 'logistic')
                    y_prob_rule = rule_model.predict_proba(filtered_features)[:, 1]
                    
                    # 檢查預測機率是否有效
                    if not np.any(np.isnan(y_prob_rule)) and not np.any(np.isinf(y_prob_rule)):
                        dca.compute_decision_curve(y, y_prob_rule, "QBRL-BC Model")
                        dca_results['rule_model'] = dca.generate_dca_report(y, y_prob_rule, "QBRL-BC Model")
                    else:
                        print("警告：QBRL-BC 模型的預測機率包含無效值，跳過DCA分析")
                except Exception as e:
                    print(f"QBRL-BC 模型DCA分析失敗：{e}")
            
            # 基準模型DCA
            try:
                trainer.train_baseline_models(X, y)
                lr_model = trainer.models['baseline_logistic_regression']
                y_prob_lr = lr_model.predict_proba(X)[:, 1]
                
                # 檢查預測機率是否有效
                if not np.any(np.isnan(y_prob_lr)) and not np.any(np.isinf(y_prob_lr)):
                    dca.compute_decision_curve(y, y_prob_lr, "Baseline Logistic Regression")
                    dca_results['baseline'] = dca.generate_dca_report(y, y_prob_lr, "Baseline Logistic Regression")
                else:
                    print("警告：基準模型的預測機率包含無效值，跳過DCA分析")
            except Exception as e:
                print(f"基準模型DCA分析失敗：{e}")
            
            # 計算參考策略
            try:
                dca.compute_reference_strategies(y)
            except Exception as e:
                print(f"參考策略計算失敗：{e}")
            
            # 保存決策曲線數據
            try:
                decisioncurve_data = dca.results
                dca_data_path = os.path.join(self.output_dir, "decision_curve_data.json")
                pd.DataFrame(decisioncurve_data).to_json(dca_data_path, orient='records', force_ascii=False, indent=2)
                print(f"決策曲線數據已保存至：{dca_data_path}")
            except Exception as e:
                print(f"決策曲線數據保存失敗：{e}")

            # 保存決策曲線圖
            try:
                plot_path = os.path.join(self.output_dir, "decision_curves.png")
                dca.plot_decision_curves(save_path=plot_path)
            except Exception as e:
                print(f"決策曲線圖繪製失敗：{e}")
            
            return dca_results
            
        except Exception as e:
            print(f"決策曲線分析失敗：{e}")
            import traceback
            print(f"詳細錯誤信息：{traceback.format_exc()}")
            return {}
    
    def _save_results(self, final_results, dca_results):
        """保存實驗結果"""
        # 更新結果字典
        self.results['final_evaluation'] = final_results
        self.results['decision_curve_analysis'] = dca_results
        self.results['timestamp'] = datetime.now().isoformat()
        
        # 保存JSON結果
        results_path = os.path.join(self.output_dir, "experiment_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # 保存文字報告
        report_path = os.path.join(self.output_dir, "experiment_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("QBRL-BC 乳癌診斷模型實驗報告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"實驗時間：{self.results['timestamp']}\n")
            f.write(f"資料集：{self.data_path}\n\n")
            
            f.write("資料信息：\n")
            for key, value in self.results['data_info'].items():
                if key != 'feature_names':
                    f.write(f"  {key}: {value}\n")
            
            f.write(f"\n最佳配置：\n")
            for key, value in self.results['best_config'].items():
                f.write(f"  {key}: {value}\n")
            
            if 'final_evaluation' in self.results:
                f.write(f"\n規則數量：{self.results['final_evaluation']['n_rules']}\n")
                f.write(f"規則特徵數量：{self.results['final_evaluation']['n_rule_features']}\n")
            
            if dca_results:
                f.write(f"\n決策曲線分析：\n")
                for model_name, report in dca_results.items():
                    f.write(f"\n{model_name}:\n{report}\n")
        
        print(f"實驗結果已保存至：{self.output_dir}")


if __name__ == "__main__":
    # 執行完整實驗
    data_path = "dataset/UCI_BCD.csv"
    
    # 創建實驗管道
    pipeline = ExperimentPipeline(
        data_path=data_path,
        output_dir="results",
        random_state=42
    )
    
    # 運行實驗
    results = pipeline.run_full_experiment()
    
    if results:
        print("\n實驗成功完成！")
        print(f"結果已保存至 'results' 目錄")
    else:
        print("\n實驗失敗！")
