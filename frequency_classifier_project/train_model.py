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
import warnings
warnings.filterwarnings('ignore')


class FrequencyClassifierTrainer:
    """
    频率分类器训练类
    """
    
    def __init__(self, data_dir: str = 'data', model_dir: str = 'model'):
        """
        初始化训练器
        
        Args:
            data_dir: 数据目录路径
            model_dir: 模型保存目录路径
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.feature_extractor = AudioFeatureExtractor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        
        # 确保模型目录存在
        os.makedirs(model_dir, exist_ok=True)
    
    def load_dataset(self, include_additional_features: bool = False) -> tuple:
        """
        加载数据集并提取特征
        
        Args:
            include_additional_features: 是否包含额外特征
            
        Returns:
            特征矩阵和标签数组
        """
        print("正在加载数据集...")
        
        features_list = []
        labels_list = []
        
        # 支持的音频格式
        audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
        
        # 遍历各个类别文件夹
        categories = ['low_freq', 'mid_freq', 'high_freq']
        
        for category in categories:
            category_path = os.path.join(self.data_dir, category)
            
            if not os.path.exists(category_path):
                print(f"警告: 类别目录 {category_path} 不存在")
                continue
            
            print(f"处理类别: {category}")
            
            # 获取该类别下的所有音频文件
            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(glob.glob(os.path.join(category_path, ext)))
            
            if not audio_files:
                print(f"警告: 在 {category_path} 中未找到音频文件")
                continue
            
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
                    print(f"处理文件 {audio_file} 时出错: {e}")
                    continue
        
        if not features_list:
            raise ValueError("未找到有效的音频文件，请检查数据目录结构")
        
        # 转换为numpy数组
        X = np.array(features_list)
        y = np.array(labels_list)
        
        print(f"\n数据集加载完成:")
        print_feature_info(X, y)
        
        return X, y
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        数据预处理
        
        Args:
            X: 特征矩阵
            y: 标签数组
            
        Returns:
            预处理后的特征矩阵和编码后的标签
        """
        print("\n正在进行数据预处理...")
        
        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 标签编码
        y_encoded = self.label_encoder.fit_transform(y)
        
        print("数据预处理完成")
        print(f"标签映射: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        return X_scaled, y_encoded
    
    def train_model(self, X: np.ndarray, y: np.ndarray, optimize_hyperparams: bool = True) -> None:
        """
        训练随机森林分类器
        
        Args:
            X: 特征矩阵
            y: 标签数组
            optimize_hyperparams: 是否进行超参数优化
        """
        print("\n正在训练模型...")
        
        if optimize_hyperparams:
            print("进行超参数优化...")
            
            # 定义超参数搜索空间
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            # 网格搜索
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
            )
            grid_search.fit(X, y)
            
            self.model = grid_search.best_estimator_
            print(f"最佳参数: {grid_search.best_params_}")
            print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")
            
        else:
            # 使用默认参数
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X, y)
        
        print("模型训练完成")
    
    def evaluate_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        评估模型性能
        
        Args:
            X: 特征矩阵
            y: 标签数组
        """
        print("\n正在评估模型性能...")
        
        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 重新训练模型（使用训练集）
        self.model.fit(X_train, y_train)
        
        # 预测
        y_pred = self.model.predict(X_test)
        
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        print(f"测试集准确率: {accuracy:.4f}")
        
        # 交叉验证
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        print(f"5折交叉验证平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # 分类报告
        print("\n分类报告:")
        target_names = self.label_encoder.classes_
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 特征重要性
        feature_importance = self.model.feature_importances_
        
        # 创建特征名称
        mfcc_names = [f'MFCC_{i+1}_mean' for i in range(13)] + [f'MFCC_{i+1}_std' for i in range(13)]
        if len(feature_importance) > 26:  # 包含额外特征
            additional_names = ['ZCR', 'Spectral_Centroid', 'Spectral_Bandwidth', 'Spectral_Rolloff', 'RMS']
            feature_names = mfcc_names + additional_names
        else:
            feature_names = mfcc_names
        
        # 绘制特征重要性
        plt.figure(figsize=(12, 8))
        indices = np.argsort(feature_importance)[::-1]
        plt.bar(range(len(feature_importance)), feature_importance[indices])
        plt.title('特征重要性')
        plt.xlabel('特征')
        plt.ylabel('重要性')
        plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self) -> None:
        """
        保存训练好的模型和预处理器
        """
        print("\n正在保存模型...")
        
        # 保存模型
        model_path = os.path.join(self.model_dir, 'frequency_classifier.pkl')
        joblib.dump(self.model, model_path)
        
        # 保存标准化器
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        # 保存标签编码器
        encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')
        joblib.dump(self.label_encoder, encoder_path)
        
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
        
        print(f"模型已保存到: {model_path}")
        print(f"标准化器已保存到: {scaler_path}")
        print(f"标签编码器已保存到: {encoder_path}")
        print(f"特征配置已保存到: {config_path}")
    
    def train_complete_pipeline(self, include_additional_features: bool = False, 
                              optimize_hyperparams: bool = True) -> None:
        """
        完整的训练流程
        
        Args:
            include_additional_features: 是否包含额外特征
            optimize_hyperparams: 是否进行超参数优化
        """
        try:
            # 1. 加载数据集
            X, y = self.load_dataset(include_additional_features)
            
            # 2. 数据预处理
            X_processed, y_processed = self.preprocess_data(X, y)
            
            # 3. 训练模型
            self.train_model(X_processed, y_processed, optimize_hyperparams)
            
            # 4. 评估模型
            self.evaluate_model(X_processed, y_processed)
            
            # 5. 保存模型
            self.save_model()
            
            print("\n训练流程完成！")
            
        except Exception as e:
            print(f"训练过程中出现错误: {e}")
            raise


def main():
    """
    主函数
    """
    print("=" * 60)
    print("音频频率分类模型训练")
    print("=" * 60)
    
    # 检查数据目录
    data_dir = 'data'
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录 {data_dir} 不存在")
        print("请确保数据目录结构如下:")
        print("data/")
        print("├── low_freq/    # 低频音频文件")
        print("├── mid_freq/    # 中频音频文件")
        print("└── high_freq/   # 高频音频文件")
        return
    
    # 创建训练器
    trainer = FrequencyClassifierTrainer(data_dir=data_dir, model_dir='model')
    
    # 配置选项
    include_additional_features = True  # 是否包含额外特征
    optimize_hyperparams = True        # 是否进行超参数优化
    
    print(f"配置:")
    print(f"  包含额外特征: {include_additional_features}")
    print(f"  超参数优化: {optimize_hyperparams}")
    print()
    
    # 开始训练
    trainer.train_complete_pipeline(
        include_additional_features=include_additional_features,
        optimize_hyperparams=optimize_hyperparams
    )


if __name__ == '__main__':
    main()