#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频频率分类模型训练脚本

功能：
1. 加载音频数据集
2. 提取MFCC特征
3. 训练RandomForest分类器
4. 评估模型性能
5. 保存训练好的模型

使用方法：
    python train_model.py
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from utils import AudioFeatureExtractor, print_feature_info
from config_manager import config, get_config
from logger_setup import get_logger, PerformanceLogger
from performance_monitor import monitor_performance, log_memory_usage
import warnings
warnings.filterwarnings('ignore')


class FrequencyClassifierTrainer:
    """
    频率分类器训练类
    """
    
    def __init__(self, data_dir: str = None, model_dir: str = None):
        """
        初始化训练器
        
        Args:
            data_dir: 数据目录路径
            model_dir: 模型保存目录路径
        """
        # 从配置文件获取默认值
        self.data_dir = data_dir or get_config('data.train_data_dir', 'data')
        self.model_dir = model_dir or get_config('data.model_dir', 'model')
        
        # 初始化日志器
        self.logger = get_logger('FrequencyClassifierTrainer')
        self.performance_logger = PerformanceLogger('training_performance')
        
        # 初始化组件
        self.feature_extractor = AudioFeatureExtractor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        
        # 确保模型目录存在
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 记录初始化信息
        self.logger.info(f"训练器初始化完成 - 数据目录: {self.data_dir}, 模型目录: {self.model_dir}")
        log_memory_usage("trainer_init")
    
    @monitor_performance("load_dataset", log_args=True)
    def load_dataset(self, include_additional_features: bool = False) -> tuple:
        """
        加载数据集并提取特征
        
        Args:
            include_additional_features: 是否包含额外特征
            
        Returns:
            特征矩阵和标签数组
        """
        self.logger.info("开始加载数据集...")
        
        features_list = []
        labels_list = []
        
        # 支持的音频格式
        audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
        
        # 从配置文件获取类别信息
        categories = get_config('frequency_classification.categories', ['low_freq', 'mid_freq', 'high_freq'])
        self.logger.debug(f"处理类别: {categories}")
        
        for category in categories:
            category_path = os.path.join(self.data_dir, category)
            
            if not os.path.exists(category_path):
                self.logger.warning(f"类别目录 {category_path} 不存在")
                continue
            
            self.logger.info(f"处理类别: {category}")
            
            # 获取该类别下的所有音频文件
            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(glob.glob(os.path.join(category_path, ext)))
            
            if not audio_files:
                self.logger.warning(f"在 {category_path} 中未找到音频文件")
                continue
            
            self.logger.debug(f"在 {category} 中找到 {len(audio_files)} 个音频文件")
            
            # 处理每个音频文件
            for audio_file in tqdm(audio_files, desc=f"处理 {category}"):
                try:
                    # 提取特征
                    features = self.feature_extractor.extract_features_from_file(
                        audio_file, include_additional_features
                    )
                    
                    features_list.append(features)
                    labels_list.append(category)
                    
                except Exception as e:
                    self.logger.error(f"处理文件 {audio_file} 时出错: {e}")
                    continue
        
        if not features_list:
            error_msg = "未找到有效的音频文件，请检查数据目录结构"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 转换为numpy数组
        X = np.array(features_list)
        y = np.array(labels_list)
        
        self.logger.info(f"数据集加载完成 - 样本数: {len(X)}, 特征维度: {X.shape[1]}, 类别数: {len(np.unique(y))}")
        print_feature_info(X, y)
        
        # 记录性能信息
        self.performance_logger.log_performance("dataset_loading", {
            "samples_count": len(X),
            "feature_dimension": X.shape[1],
            "categories_count": len(np.unique(y)),
            "include_additional_features": include_additional_features
        })
        
        return X, y
    
    @monitor_performance("preprocess_data")
    def preprocess_data(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        数据预处理
        
        Args:
            X: 特征矩阵
            y: 标签数组
            
        Returns:
            预处理后的特征矩阵和编码后的标签
        """
        self.logger.info("开始数据预处理...")
        
        # 特征标准化
        self.logger.debug(f"特征标准化前 - 形状: {X.shape}, 均值: {X.mean():.4f}, 标准差: {X.std():.4f}")
        X_scaled = self.scaler.fit_transform(X)
        self.logger.debug(f"特征标准化后 - 均值: {X_scaled.mean():.4f}, 标准差: {X_scaled.std():.4f}")
        
        # 标签编码
        unique_labels = np.unique(y)
        self.logger.debug(f"原始标签: {unique_labels}")
        y_encoded = self.label_encoder.fit_transform(y)
        
        label_mapping = dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))
        self.logger.info(f"数据预处理完成 - 标签映射: {label_mapping}")
        print(f"标签映射: {label_mapping}")
        
        # 记录性能信息
        self.performance_logger.log_performance("data_preprocessing", {
            "input_shape": X.shape,
            "output_shape": X_scaled.shape,
            "unique_labels_count": len(unique_labels),
            "label_mapping": label_mapping
        })
        
        return X_scaled, y_encoded
    
    @monitor_performance("train_model", log_args=True)
    def train_model(self, X: np.ndarray, y: np.ndarray, optimize_hyperparams: bool = True) -> None:
        """
        训练随机森林分类器
        
        Args:
            X: 特征矩阵
            y: 标签数组
            optimize_hyperparams: 是否进行超参数优化
        """
        self.logger.info("开始训练模型...")
        
        if optimize_hyperparams:
            self.logger.info("进行超参数优化...")
            
            # 从配置文件获取超参数搜索空间
            param_grid = get_config('model.hyperparameter_tuning.param_grid', {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            })
            self.logger.debug(f"超参数搜索空间: {param_grid}")
            
            # 网格搜索
            rf = RandomForestClassifier(random_state=42)
            cv_folds = get_config('model.hyperparameter_tuning.cv_folds', 5)
            grid_search = GridSearchCV(
                rf, param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1, verbose=1
            )
            
            self.logger.info(f"开始网格搜索，CV折数: {cv_folds}")
            grid_search.fit(X, y)
            
            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            self.logger.info(f"超参数优化完成 - 最佳参数: {best_params}, 最佳得分: {best_score:.4f}")
            print(f"最佳参数: {best_params}")
            print(f"最佳交叉验证得分: {best_score:.4f}")
            
            # 记录超参数优化结果
            self.performance_logger.log_performance("hyperparameter_optimization", {
                "best_params": best_params,
                "best_score": best_score,
                "cv_folds": cv_folds
            })
            
        else:
            # 从配置文件获取默认参数
            default_params = get_config('model.random_forest', {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'random_state': 42,
                'n_jobs': -1
            })
            
            self.logger.debug(f"使用默认参数: {default_params}")
            self.model = RandomForestClassifier(**default_params)
            self.model.fit(X, y)
        
        self.logger.info("模型训练完成")
        
        # 记录训练信息
        self.performance_logger.log_performance("model_training", {
            "training_samples": len(X),
            "feature_dimension": X.shape[1],
            "optimize_hyperparams": optimize_hyperparams,
            "model_type": "RandomForestClassifier"
        })
    
    @monitor_performance("evaluate_model")
    def evaluate_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        评估模型性能
        
        Args:
            X: 特征矩阵
            y: 标签数组
        """
        self.logger.info("开始评估模型性能...")
        
        # 从配置获取测试集比例
        test_size = get_config('model.evaluation.test_size', 0.2)
        cv_folds = get_config('model.evaluation.cv_folds', 5)
        
        # 分割数据集
        self.logger.debug(f"分割数据集，测试集比例: {test_size}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        self.logger.debug(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        
        # 重新训练模型（使用训练集）
        self.model.fit(X_train, y_train)
        
        # 预测
        y_pred = self.model.predict(X_test)
        
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        self.logger.info(f"测试集准确率: {accuracy:.4f}")
        print(f"测试集准确率: {accuracy:.4f}")
        
        # 交叉验证
        self.logger.debug(f"进行{cv_folds}折交叉验证")
        cv_scores = cross_val_score(self.model, X, y, cv=cv_folds)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        self.logger.info(f"{cv_folds}折交叉验证结果 - 平均准确率: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
        print(f"{cv_folds}折交叉验证平均准确率: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
        
        # 分类报告
        print("\n分类报告:")
        target_names = self.label_encoder.classes_
        class_report = classification_report(y_test, y_pred, target_names=target_names)
        print(class_report)
        self.logger.debug(f"分类报告:\n{class_report}")
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        self.logger.debug(f"混淆矩阵:\n{cm}")
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        
        confusion_matrix_path = os.path.join(self.model_dir, 'confusion_matrix.png')
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"混淆矩阵已保存到: {confusion_matrix_path}")
        plt.show()
        
        # 特征重要性
        feature_importance = self.model.feature_importances_
        self.logger.debug(f"特征重要性形状: {feature_importance.shape}")
        
        # 创建特征名称
        n_mfcc = get_config('audio_processing.n_mfcc', 13)
        mfcc_names = [f'MFCC_{i+1}_mean' for i in range(n_mfcc)] + [f'MFCC_{i+1}_std' for i in range(n_mfcc)]
        if len(feature_importance) > n_mfcc * 2:  # 包含额外特征
            additional_names = ['ZCR', 'Spectral_Centroid', 'Spectral_Bandwidth', 'Spectral_Rolloff', 'RMS']
            feature_names = mfcc_names + additional_names
        else:
            feature_names = mfcc_names
        
        # 记录最重要的特征
        indices = np.argsort(feature_importance)[::-1]
        top_features = [(feature_names[i], feature_importance[i]) for i in indices[:5]]
        self.logger.info(f"前5个最重要特征: {top_features}")
        
        # 绘制特征重要性
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(feature_importance)), feature_importance[indices])
        plt.title('特征重要性')
        plt.xlabel('特征')
        plt.ylabel('重要性')
        plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        
        feature_importance_path = os.path.join(self.model_dir, 'feature_importance.png')
        plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"特征重要性图已保存到: {feature_importance_path}")
        plt.show()
        
        # 记录评估性能信息
        self.performance_logger.log_performance("model_evaluation", {
            "test_accuracy": accuracy,
            "cv_mean_accuracy": cv_mean,
            "cv_std_accuracy": cv_std,
            "test_size": test_size,
            "cv_folds": cv_folds,
            "confusion_matrix": cm.tolist(),
            "top_features": top_features
        })
    
    @monitor_performance("save_model")
    def save_model(self) -> None:
        """
        保存训练好的模型和预处理器
        """
        self.logger.info("开始保存模型...")
        
        # 保存模型
        model_path = os.path.join(self.model_dir, 'frequency_classifier.pkl')
        joblib.dump(self.model, model_path)
        self.logger.debug(f"模型已保存到: {model_path}")
        
        # 保存标准化器
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        self.logger.debug(f"标准化器已保存到: {scaler_path}")
        
        # 保存标签编码器
        encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')
        joblib.dump(self.label_encoder, encoder_path)
        self.logger.debug(f"标签编码器已保存到: {encoder_path}")
        
        # 保存特征提取器配置
        config = {
            'sample_rate': self.feature_extractor.sample_rate,
            'n_mfcc': self.feature_extractor.n_mfcc,
            'n_fft': self.feature_extractor.n_fft,
            'hop_length': self.feature_extractor.hop_length,
            'duration': self.feature_extractor.duration
        }
        config_path = os.path.join(self.model_dir, 'feature_config.pkl')
        joblib.dump(config, config_path)
        self.logger.debug(f"特征配置已保存到: {config_path}")
        
        # 计算文件大小
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        
        self.logger.info(f"模型保存完成 - 模型大小: {model_size:.2f}MB")
        print(f"模型已保存到: {model_path}")
        print(f"标准化器已保存到: {scaler_path}")
        print(f"标签编码器已保存到: {encoder_path}")
        print(f"特征配置已保存到: {config_path}")
        
        # 记录保存信息
        self.performance_logger.log_performance("model_saving", {
            "model_path": model_path,
            "model_size_mb": model_size,
            "scaler_path": scaler_path,
            "encoder_path": encoder_path,
            "config_path": config_path
        })
    
    @monitor_performance("train_complete_pipeline", log_args=True)
    def train_complete_pipeline(self, include_additional_features: bool = False, 
                              optimize_hyperparams: bool = True) -> None:
        """
        完整的训练流程
        
        Args:
            include_additional_features: 是否包含额外特征
            optimize_hyperparams: 是否进行超参数优化
        """
        self.logger.info("开始完整训练流程...")
        
        try:
            # 1. 加载数据集
            self.logger.info("步骤 1/5: 加载数据集")
            X, y = self.load_dataset(include_additional_features)
            
            # 2. 数据预处理
            self.logger.info("步骤 2/5: 数据预处理")
            X_processed, y_processed = self.preprocess_data(X, y)
            
            # 3. 训练模型
            self.logger.info("步骤 3/5: 训练模型")
            self.train_model(X_processed, y_processed, optimize_hyperparams)
            
            # 4. 评估模型
            self.logger.info("步骤 4/5: 评估模型")
            self.evaluate_model(X_processed, y_processed)
            
            # 5. 保存模型
            self.logger.info("步骤 5/5: 保存模型")
            self.save_model()
            
            self.logger.info("完整训练流程成功完成！")
            print("\n训练流程完成！")
            
            # 记录整体训练性能
            self.performance_logger.log_performance("complete_training_pipeline", {
                "include_additional_features": include_additional_features,
                "optimize_hyperparams": optimize_hyperparams,
                "total_samples": len(X),
                "feature_dimension": X.shape[1],
                "success": True
            })
            
        except Exception as e:
            error_msg = f"训练过程中出现错误: {e}"
            self.logger.error(error_msg)
            print(error_msg)
            
            # 记录失败信息
            self.performance_logger.log_performance("complete_training_pipeline", {
                "include_additional_features": include_additional_features,
                "optimize_hyperparams": optimize_hyperparams,
                "success": False,
                "error": str(e)
            })
            
            raise


def main():
    """
    主函数
    """
    # 初始化日志器
    logger = get_logger('main')
    
    print("=" * 60)
    print("音频频率分类模型训练")
    print("=" * 60)
    
    logger.info("开始音频频率分类模型训练")
    
    # 从配置文件获取目录路径
    data_dir = get_config('data.train_data_dir', 'data')
    model_dir = get_config('data.model_dir', 'model')
    
    # 检查数据目录
    if not os.path.exists(data_dir):
        error_msg = f"错误: 数据目录 {data_dir} 不存在"
        logger.error(error_msg)
        print(error_msg)
        print("请确保数据目录结构如下:")
        print("data/")
        print("├── low_freq/    # 低频音频文件")
        print("├── mid_freq/    # 中频音频文件")
        print("└── high_freq/   # 高频音频文件")
        return
    
    logger.info(f"数据目录检查通过: {data_dir}")
    
    # 创建训练器
    trainer = FrequencyClassifierTrainer(data_dir=data_dir, model_dir=model_dir)
    
    # 从配置文件获取训练选项
    include_additional_features = get_config('model.training.include_additional_features', True)
    optimize_hyperparams = get_config('model.training.optimize_hyperparams', True)
    
    logger.info(f"训练配置 - 包含额外特征: {include_additional_features}, 超参数优化: {optimize_hyperparams}")
    
    print(f"配置:")
    print(f"  数据目录: {data_dir}")
    print(f"  模型目录: {model_dir}")
    print(f"  包含额外特征: {include_additional_features}")
    print(f"  超参数优化: {optimize_hyperparams}")
    print()
    
    # 开始训练
    try:
        trainer.train_complete_pipeline(
            include_additional_features=include_additional_features,
            optimize_hyperparams=optimize_hyperparams
        )
        logger.info("训练任务成功完成")
    except Exception as e:
        logger.error(f"训练任务失败: {e}")
        raise


if __name__ == '__main__':
    main()